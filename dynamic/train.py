"""Training: sampled softmax over fixed corruptions, model selection on withheld entities."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import Dataset
from .evaluate import KnownAnswers, RankingTask
from .model import DyNAMiC, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters; defaults as in the paper."""

    num_negatives: int = 16  # corruptions per training fact [n_neg]
    lr: float = 5e-4
    batch_size: int = 256
    epochs: int = 60
    patience: int = 8  # epochs without a better validation MRR before stopping
    valid_candidates: int = 200  # negatives per direction in validation
    seed: int = 42  # model initialisation and batch order


def corrupt(facts: Tensor, num_entities: int, k: int, seed: int = 42) -> tuple[Tensor, Tensor]:
    """``k`` corruptions per fact, as (E, k) heads and (E, k) tails.

    Each corruption replaces the head or the tail, with equal probability, by an entity
    drawn uniformly from ``range(num_entities)``, without filtering.
    """
    gen = torch.Generator().manual_seed(seed)
    replacement = torch.randint(0, num_entities, (len(facts), k), generator=gen)
    replace_head = torch.rand(len(facts), k, generator=gen) < 0.5
    h, t = facts[:, :1], facts[:, 2:]
    return torch.where(replace_head, replacement, h), torch.where(replace_head, t, replacement)


def train(
    dataset: Dataset,
    model_config: ModelConfig = ModelConfig(),
    config: TrainConfig = TrainConfig(),
    device: torch.device | str = "cpu",
    log: Callable[[str], None] = print,
) -> tuple[DyNAMiC, list[dict]]:
    """Train on ``dataset.train`` and return the model of the best validation epoch.

    After every epoch the model is bound to the inference graph and ranks the validation
    queries against ``config.valid_candidates`` negatives per direction. Training stops
    after ``config.patience`` epochs without a better MRR. Also returns one record per
    epoch.
    """
    torch.manual_seed(config.seed)
    model = DyNAMiC(dataset.num_seen, dataset.num_relations, model_config).to(device)
    train_facts = model.sample_facts(dataset.train, dataset.num_seen, device)
    inference_facts = model.sample_facts(dataset.inference, dataset.num_entities, device)
    known = KnownAnswers(dataset.inference, dataset.valid, dataset.test)
    validation = RankingTask.sample(
        dataset.valid, known, dataset.num_entities, config.valid_candidates
    )

    facts = dataset.train.to(device)
    neg_heads, neg_tails = (
        x.to(device) for x in corrupt(dataset.train, dataset.num_seen, config.num_negatives)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[dict] = []
    best_mrr, best_state, stale = float("-inf"), None, 0
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        model.train()
        batches = torch.randperm(len(facts)).split(config.batch_size)
        total_loss = torch.zeros((), device=device)
        for batch in batches:
            batch = batch.to(device)
            h, r, t = facts[batch].unbind(dim=1)
            # one softmax group per fact: the fact itself (column 0), then its corruptions
            heads = torch.cat([h.unsqueeze(1), neg_heads[batch]], dim=1)
            tails = torch.cat([t.unsqueeze(1), neg_tails[batch]], dim=1)
            n = heads.shape[1]
            scores = model(
                train_facts,
                heads.flatten(),
                r.repeat_interleave(n),
                tails.flatten(),
                anchor=(h.repeat_interleave(n), t.repeat_interleave(n)),
            ).view_as(heads)
            loss = F.cross_entropy(scores, torch.zeros_like(h))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()

        valid = validation.evaluate(model, inference_facts)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss.item() / len(batches),
                "valid": valid,
                "seconds": round(time.time() - start, 1),
            }
        )
        log(
            f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  valid MRR {valid['mrr']:.4f}  "
            f"H@10 {valid['hits@10']:.4f}  ({history[-1]['seconds']:.0f}s)"
        )
        if valid["mrr"] > best_mrr:
            best_mrr, best_state, stale = valid["mrr"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, history
