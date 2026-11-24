"""DyNAMiC: scores a triple (h, r, t) from the facts around h and t.

Each entity is represented by a buffer of K sampled facts (neighbour, relation). A fact
is encoded as ``[e(v); w_rho]``, where ``e(v)`` is a learned embedding for a seen
neighbour and a shared unknown vector otherwise. The facts of h and of t attend to each
other through L layers of differential attention, each side is mean-pooled, and an MLP
scores ``[z_h; z_t; q_r]``. The model never embeds h or t themselves, so it can score
entities withheld from training as long as they have facts at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

#: Neighbour id of an empty buffer slot.
PAD = -1


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of DyNAMiC (paper notation in brackets); defaults as in the paper."""

    num_facts: int = 16  # facts buffered per entity [K]
    entity_dim: int = 64  # [d_e]
    relation_dim: int = 64  # [d_r]
    query_dim: int = 64  # query-relation embedding [d_q]
    num_layers: int = 2  # [L]
    num_heads: int = 8  # [n_h]
    ffn_dim: int = 64
    dropout: float = 0.1  # in the attention
    identity_dropout: float = 0.3  # [p_id]
    decoder_layers: int = 4
    decoder_dim: int = 512
    relation_only: bool = False  # describe every neighbour by its relation alone


@dataclass(frozen=True)
class FactBuffer:
    """K facts per entity of one graph: ``neighbours`` and ``relations``, both (V, K).

    Every fact (u, r, v) gives u the fact (v, r) and v the inverse fact (u, r + R). An
    entity with more than K facts keeps a seeded random subset; empty slots hold the
    padding fact (``PAD``, 2R).
    """

    neighbours: Tensor
    relations: Tensor

    @classmethod
    def sample(
        cls,
        facts: Tensor,
        num_entities: int,
        num_relations: int,
        k: int,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> FactBuffer:
        h, r, t = facts.to(device).unbind(dim=1)
        src, dst, rel = torch.cat([h, t]), torch.cat([t, h]), torch.cat([r, r + num_relations])
        gen = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(len(src), generator=gen, device=device)
        order = perm[torch.argsort(src[perm], stable=True)]  # grouped by entity, shuffled within
        src, dst, rel = src[order], dst[order], rel[order]
        count = torch.bincount(src, minlength=num_entities)
        slot = torch.arange(len(src), device=device) - (count.cumsum(0) - count)[src]
        keep = slot < k
        neighbours = torch.full((num_entities, k), PAD, device=device)
        relations = torch.full((num_entities, k), 2 * num_relations, device=device)
        neighbours[src[keep], slot[keep]] = dst[keep]
        relations[src[keep], slot[keep]] = rel[keep]
        return cls(neighbours, relations)


class FactEncoder(nn.Module):
    """Encodes facts as ``[e(v); w_rho]`` (Eqs. 1 and 2 of the paper).

    ``e(v)`` is the learned row of a seen neighbour (id below ``num_seen``) and the
    unknown vector for a withheld neighbour or padding. Identity dropout replaces each
    seen neighbour by the unknown vector with probability ``identity_dropout`` during
    training. ``relation_only`` uses the unknown vector for every neighbour.
    """

    def __init__(
        self,
        num_seen: int,
        num_relations: int,
        entity_dim: int,
        relation_dim: int,
        identity_dropout: float,
        relation_only: bool,
    ) -> None:
        super().__init__()
        self.num_seen = num_seen
        self.identity_dropout = identity_dropout
        self.entity = None if relation_only else nn.Embedding(num_seen, entity_dim)
        self.unknown = nn.Parameter(torch.empty(1, entity_dim))
        self.relation = nn.Embedding(2 * num_relations + 1, relation_dim)  # last row: padding
        nn.init.xavier_uniform_(self.unknown)
        nn.init.xavier_uniform_(self.relation.weight)
        if self.entity is not None:
            nn.init.xavier_uniform_(self.entity.weight)

    def forward(self, neighbours: Tensor, relations: Tensor) -> Tensor:
        unknown = self.unknown[0].expand(*neighbours.shape, -1)
        if self.entity is None:
            node = unknown
        else:
            known = (neighbours >= 0) & (neighbours < self.num_seen)
            if self.training and self.identity_dropout > 0:
                known &= torch.rand(known.shape, device=known.device) >= self.identity_dropout
            entity = self.entity(neighbours.clamp(0, self.num_seen - 1))
            node = torch.where(known.unsqueeze(-1), entity, unknown)
        return torch.cat([node, self.relation(relations)], dim=-1)


class DifferentialAttention(nn.Module):
    """Multi-head differential attention (Ye et al., 2025).

    Each head subtracts two softmax maps, ``A1 - lambda * A2``, computed from separate
    query and key projections of width ``d / num_heads``; its values have twice that
    width. Head outputs are RMS-normalised, scaled by ``1 - lambda_init`` and projected
    back to ``d``. ``mask`` is added to the attention logits.
    """

    #: Ye et al.'s depth rule at layer index 1, used for every layer.
    LAMBDA_INIT = 0.8 - 0.6 * math.exp(-0.3)

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, 2 * dim, bias=False)
        self.key = nn.Linear(dim, 2 * dim, bias=False)
        self.value = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(2 * dim, dim, bias=False)
        # lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init, one per head
        self.lambda_q = nn.Parameter(0.1 * torch.randn(2, num_heads, self.head_dim))
        self.lambda_k = nn.Parameter(0.1 * torch.randn(2, num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """``x`` (B, Nx, d) attends over ``y`` (B, Ny, d); ``mask`` is (B, Nx, Ny)."""
        b, nx, ny, h, dk = x.shape[0], x.shape[1], y.shape[1], self.num_heads, self.head_dim
        q = self.query(x).view(b, nx, h, 2, dk).permute(3, 0, 2, 1, 4)  # (2, B, H, Nx, dk)
        k = self.key(y).view(b, ny, h, 2, dk).permute(3, 0, 2, 1, 4)
        v = self.value(y).view(b, ny, h, 2 * dk).transpose(1, 2)  # (B, H, Ny, 2dk)
        a1, a2 = (q @ k.transpose(-1, -2) * dk**-0.5 + mask.unsqueeze(1)).softmax(dim=-1)
        lam = (self.lambda_q * self.lambda_k).sum(dim=-1).exp()
        lam = (lam[0] - lam[1] + self.LAMBDA_INIT).view(h, 1, 1)
        heads = self.dropout(a1 - lam * a2) @ v
        heads = heads * torch.rsqrt(heads.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        heads = heads.transpose(1, 2).reshape(b, nx, 2 * h * dk) * (1 - self.LAMBDA_INIT)
        return self.dropout(self.out(heads))


class CrossAttentionLayer(nn.Module):
    """One layer of Eqs. 3 and 4: the facts of h attend over those of t and vice versa.

    The attention module is shared by the two directions; the feed-forward networks and
    layer norms are not.
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = DifferentialAttention(dim, num_heads, dropout)
        self.norm_h1, self.norm_t1 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn_h, self.ffn_t = _ffn(dim, ffn_dim), _ffn(dim, ffn_dim)
        self.norm_h2, self.norm_t2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, h: Tensor, t: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``h`` (B, K, d), ``t`` (B, K, d); ``mask`` (B, K, K) for h's facts over t's."""
        h, t = (
            self.norm_h1(h + self.attention(h, t, mask)),
            self.norm_t1(t + self.attention(t, h, mask.transpose(1, 2))),
        )
        return self.norm_h2(h + self.ffn_h(h)), self.norm_t2(t + self.ffn_t(t))


def _ffn(dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))


class DyNAMiC(nn.Module):
    """The model, for a training graph of ``num_seen`` entities and ``num_relations``
    relation types (inverses not counted)."""

    def __init__(
        self, num_seen: int, num_relations: int, config: ModelConfig = ModelConfig()
    ) -> None:
        super().__init__()
        c = config
        dim = c.entity_dim + c.relation_dim
        if dim % c.num_heads:
            raise ValueError(
                f"num_heads ({c.num_heads}) must divide entity_dim + relation_dim ({dim})."
            )
        self.config = config
        self.num_relations = num_relations
        self.facts = FactEncoder(
            num_seen,
            num_relations,
            c.entity_dim,
            c.relation_dim,
            c.identity_dropout,
            c.relation_only,
        )
        self.query_relation = nn.Embedding(num_relations, c.query_dim)
        nn.init.xavier_uniform_(self.query_relation.weight)
        self.layers = nn.ModuleList(
            CrossAttentionLayer(dim, c.num_heads, c.ffn_dim, c.dropout) for _ in range(c.num_layers)
        )
        decoder = [nn.Linear(2 * dim + c.query_dim, c.decoder_dim), nn.ReLU()]
        for _ in range(c.decoder_layers - 1):
            decoder += [nn.Linear(c.decoder_dim, c.decoder_dim), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder, nn.Linear(c.decoder_dim, 1))

    def sample_facts(
        self, facts: Tensor, num_entities: int, device: torch.device | str
    ) -> FactBuffer:
        """The fact buffer of a graph, sized for this model."""
        return FactBuffer.sample(
            facts, num_entities, self.num_relations, self.config.num_facts, device=device
        )

    def forward(
        self,
        buffer: FactBuffer,
        h: Tensor,
        r: Tensor,
        t: Tensor,
        anchor: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Scores of the triples ``(h[i], r[i], t[i])`` over the graph of ``buffer``.

        ``anchor`` is the (head, tail) of the training fact each row belongs to. Its edge
        is hidden from both endpoints' facts, for the fact and for its corruptions alike
        (anchor masking). Leave it unset at inference, where query facts are not in the
        graph.
        """
        h_nb, h_rel = buffer.neighbours[h], buffer.relations[h]
        t_nb, t_rel = buffer.neighbours[t], buffer.relations[t]
        if anchor is not None:
            anchor_h, anchor_t = anchor
            h_nb, h_rel = self._hide(h_nb, h_rel, anchor_t, r)
            t_nb, t_rel = self._hide(t_nb, t_rel, anchor_h, r + self.num_relations)
        x_h, x_t = self.facts(h_nb, h_rel), self.facts(t_nb, t_rel)
        padding = (h_nb == PAD).unsqueeze(2) | (t_nb == PAD).unsqueeze(1)
        mask = torch.zeros(padding.shape, device=padding.device).masked_fill_(padding, -1e9)
        for layer in self.layers:
            x_h, x_t = layer(x_h, x_t, mask)
        z = torch.cat([x_h.mean(dim=1), x_t.mean(dim=1), self.query_relation(r)], dim=-1)
        return self.decoder(z).squeeze(-1)

    def _hide(
        self, neighbours: Tensor, relations: Tensor, neighbour: Tensor, relation: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Turn the slots holding the fact (neighbour[i], relation[i]) into padding."""
        hit = (neighbours == neighbour.unsqueeze(1)) & (relations == relation.unsqueeze(1))
        return neighbours.masked_fill(hit, PAD), relations.masked_fill(hit, 2 * self.num_relations)
