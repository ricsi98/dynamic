"""The ranking protocol of the paper.

Each query (h, r, t) is asked in both directions. The tail query ranks t against C
negatives drawn uniformly without replacement from all entities, excluding every known
answer of (h, r, ?); the head query is treated symmetrically. The rank is pessimistic:
one plus the number of negatives scoring at least as high as the answer. The draw is
seeded, so every model is ranked against the same candidates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import Dataset
from .model import DyNAMiC, FactBuffer


class KnownAnswers:
    """The known tails of every (h, r) and heads of every (r, t) among some facts."""

    def __init__(self, *facts: Tensor) -> None:
        self.tails: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.heads: dict[tuple[int, int], set[int]] = defaultdict(set)
        for split in facts:
            for h, r, t in split.tolist():
                self.tails[h, r].add(t)
                self.heads[r, t].add(h)


@dataclass(frozen=True)
class RankingTask:
    """Queries (Q, 3) with their tail and head negatives, both (Q, C)."""

    queries: Tensor
    tail_candidates: Tensor
    head_candidates: Tensor

    @classmethod
    def sample(
        cls,
        queries: Tensor,
        known: KnownAnswers,
        num_entities: int,
        num_candidates: int,
        seed: int = 42,
    ) -> RankingTask:
        draws = _uniform_stream(num_entities, max(4096, 4 * num_candidates), seed)
        tails, heads = [], []
        for h, r, t in queries.tolist():
            tail_excluded = known.tails.get((h, r), set()) | {t}
            head_excluded = known.heads.get((r, t), set()) | {h}
            tails.append(_distinct(draws, tail_excluded, num_candidates, num_entities))
            heads.append(_distinct(draws, head_excluded, num_candidates, num_entities))
        shape = (len(queries), num_candidates)
        return cls(queries, torch.tensor(tails).view(shape), torch.tensor(heads).view(shape))

    @torch.no_grad()
    def evaluate(
        self, model: DyNAMiC, facts: FactBuffer, max_pairs: int = 16384
    ) -> dict[str, float]:
        """MRR and Hits@{1,3,10} over both directions; ``max_pairs`` bounds memory."""
        was_training = model.training
        model.eval()
        ranks = torch.cat([self.ranks(model, facts, tail, max_pairs) for tail in (True, False)])
        model.train(was_training)
        return {
            "mrr": (1 / ranks.float()).mean().item(),
            **{f"hits@{k}": (ranks <= k).float().mean().item() for k in (1, 3, 10)},
        }

    def ranks(
        self, model: DyNAMiC, facts: FactBuffer, predict_tail: bool, max_pairs: int
    ) -> Tensor:
        """Rank of the answer of every tail (or head) query."""
        candidates = self.tail_candidates if predict_tail else self.head_candidates
        device = facts.neighbours.device
        step = max(1, max_pairs // (1 + candidates.shape[1]))
        ranks = []
        for queries, negatives in zip(self.queries.split(step), candidates.split(step)):
            h, r, t = queries.to(device).unbind(dim=1)
            answer, given = (t, h) if predict_tail else (h, t)
            # the answer and its negatives are scored in one pass, so exact ties stay ties
            entities = torch.cat([answer.unsqueeze(1), negatives.to(device)], dim=1)
            n = entities.shape[1]
            given, r = given.repeat_interleave(n), r.repeat_interleave(n)
            heads, tails = (
                (given, entities.flatten()) if predict_tail else (entities.flatten(), given)
            )
            scores = model(facts, heads, r, tails).view_as(entities)
            scores = scores.nan_to_num(nan=float("-inf"))  # a diverged model ranks last, not first
            ranks.append(1 + (scores[:, 1:] >= scores[:, :1]).sum(dim=1).cpu())
        return torch.cat(ranks)


def evaluate(
    model: DyNAMiC, dataset: Dataset, split: str = "test", num_candidates: int = 500, seed: int = 42
) -> dict[str, float]:
    """Rank the ``valid`` or ``test`` queries of ``dataset`` over its inference graph."""
    device = next(model.parameters()).device
    facts = model.sample_facts(dataset.inference, dataset.num_entities, device)
    known = KnownAnswers(dataset.inference, dataset.valid, dataset.test)
    queries = getattr(dataset, split)
    task = RankingTask.sample(queries, known, dataset.num_entities, num_candidates, seed)
    return task.evaluate(model, facts)


def _uniform_stream(n: int, chunk: int, seed: int) -> Iterator[int]:
    gen = torch.Generator().manual_seed(seed)
    while True:
        yield from torch.randint(0, n, (chunk,), generator=gen).tolist()


def _distinct(draws: Iterator[int], excluded: set[int], k: int, n: int) -> list[int]:
    """The next ``k`` distinct draws from ``range(n)`` that are not in ``excluded``."""
    if n - len(excluded) < k:
        raise ValueError(f"{k} negatives requested, but only {n - len(excluded)} entities qualify.")
    taken, out = set(excluded), []
    while len(out) < k:
        c = next(draws)
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out
