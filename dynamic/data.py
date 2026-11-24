"""The semi-inductive splits of the paper: FB15k-237-semi, WN18RR-semi and OOKB.

Every split has a training graph over the *seen* entities and a larger inference graph
that adds the *withheld* entities and their facts. Seen entities have ids
``[0, num_seen)`` in both graphs, so their trained embeddings apply at inference, and
withheld entities follow at ``[num_seen, num_entities)``. Test queries involve at least
one withheld entity, and so do validation queries except on OOKB, whose official dev
facts mostly involve seen entities only. Queries are disjoint from the inference graph.

Facts are ``(E, 3)`` integer tensors of ``(head, relation, tail)`` rows. Raw files are
downloaded on first use into ``root``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

_KGE_URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master"
_OOKB_URL = "https://raw.githubusercontent.com/takuo-h/GNN-for-OOKB/master/0-OOKB-raw-datasets"

DATASETS = ("FB15k-237-semi", "WN18RR-semi", "OOKB-1000", "OOKB-3000", "OOKB-5000")

#: Seed of every random choice made while building a split.
SPLIT_SEED = 42


@dataclass(frozen=True)
class Dataset:
    """One split; facts and queries are (E, 3) rows of (head, relation, tail)."""

    name: str
    num_seen: int  # entities of the training graph
    num_entities: int  # entities of the inference graph (seen + withheld)
    num_relations: int  # relation types, inverses not counted
    train: Tensor  # facts over seen entities
    inference: Tensor  # facts over all entities, a superset of ``train``
    valid: Tensor  # validation queries
    test: Tensor  # test queries, each with a withheld endpoint

    def __post_init__(self) -> None:
        graph = {tuple(f) for f in self.inference.tolist()}
        for split in ("valid", "test"):
            if any(tuple(q) in graph for q in getattr(self, split).tolist()):
                raise ValueError(f"{self.name}: {split} queries overlap the inference graph.")


def load_dataset(name: str, root: str | Path = "data") -> Dataset:
    """Load one of :data:`DATASETS`, downloading its raw files if needed."""
    root = Path(root)
    if name == "FB15k-237-semi":
        return _withhold_entities(name, *_fb15k237(root / "FB15k-237"))
    if name == "WN18RR-semi":
        return _withhold_entities(name, *_wn18rr(root / "WN18RR"))
    if name.startswith("OOKB-") and name[5:] in ("1000", "3000", "5000"):
        return _ookb(name, root / name)
    raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}.")


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _read(url: str, path: Path) -> list[list[str]]:
    """Tab-separated rows of a raw file, downloading it first if needed."""
    lines = _download(url, path).read_text().splitlines()
    return [line.strip().split("\t") for line in lines if line.strip()]


def _fb15k237(root: Path) -> tuple[list[Tensor], int, int]:
    """FB15k-237 train/valid/test; ids in order of first appearance across the files."""
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}
    splits = []
    for split in ("train", "valid", "test"):
        rows = _read(f"{_KGE_URL}/FB15k-237/{split}.txt", root / f"{split}.txt")
        splits.append(
            torch.tensor(
                [
                    [
                        entities.setdefault(h, len(entities)),
                        relations.setdefault(r, len(relations)),
                        entities.setdefault(t, len(entities)),
                    ]
                    for h, r, t in rows
                ]
            )
        )
    return splits, len(entities), len(relations)


def _wn18rr(root: Path) -> tuple[list[Tensor], int, int]:
    """WN18RR train/valid/test, each ordered by (head, tail).

    Entity ids follow first appearance among each file's heads, then its tails; relation
    ids follow sorted names. These are the ids and the order of the splits in the paper.
    """
    entities: dict[str, int] = {}
    rows = [
        _read(f"{_KGE_URL}/WN18RR/original/{split}.txt", root / f"{split}.txt")
        for split in ("train", "valid", "test")
    ]
    relations = {r: i for i, r in enumerate(sorted({r for split in rows for _, r, _ in split}))}
    splits = []
    for split in rows:
        for e in [h for h, _, _ in split] + [t for _, _, t in split]:
            entities.setdefault(e, len(entities))
        splits.append(torch.tensor([[entities[h], relations[r], entities[t]] for h, r, t in split]))
    n = len(entities)
    order = [torch.sort(s[:, 0] * n + s[:, 2], stable=True).indices for s in splits]
    return [s[o] for s, o in zip(splits, order)], n, len(relations)


def _withhold_entities(
    name: str, splits: list[Tensor], num_entities: int, num_relations: int, fraction: float = 0.1
) -> Dataset:
    """Withhold a random ``fraction`` of the entities of a transductive dataset.

    The training graph keeps the training facts between seen entities; the inference
    graph is the full training graph. Queries are the validation and test facts with a
    withheld endpoint, each of which has a training fact linking it to a seen entity.
    """
    train, valid, test = splits
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    withheld = torch.zeros(num_entities, dtype=torch.bool)
    withheld[torch.randperm(num_entities, generator=gen)[: int(fraction * num_entities)]] = True
    num_seen = int((~withheld).sum())
    new_id = torch.empty(num_entities, dtype=torch.long)
    new_id[~withheld] = torch.arange(num_seen)
    new_id[withheld] = torch.arange(num_seen, num_entities)

    def relabel(facts: Tensor) -> Tensor:
        return torch.stack([new_id[facts[:, 0]], facts[:, 1], new_id[facts[:, 2]]], dim=1)

    head_out, tail_out = withheld[train[:, 0]], withheld[train[:, 2]]
    linked = torch.zeros(num_entities, dtype=torch.bool)  # withheld, with a seen neighbour
    linked[train[head_out & ~tail_out, 0]] = True
    linked[train[tail_out & ~head_out, 2]] = True

    def queries(facts: Tensor) -> Tensor:
        h, t = facts[:, 0], facts[:, 2]
        keep = (withheld[h] | withheld[t]) & (~withheld[h] | linked[h]) & (~withheld[t] | linked[t])
        return relabel(facts[keep])

    return Dataset(
        name,
        num_seen,
        num_entities,
        num_relations,
        train=relabel(train[~head_out & ~tail_out]),
        inference=relabel(train),
        valid=queries(valid),
        test=queries(test),
    )


def _ookb(name: str, root: Path) -> Dataset:
    """An OOKB split (Hamaguchi et al., 2017), ``both`` variant, posed as ranking.

    Seen entities are those of the ``train`` file. The inference graph adds the ``aux``
    facts that attach the withheld entities; the queries are the true (+1) facts of the
    ``dev`` and ``test`` files. A seeded 10% of the ``train`` facts is left out of the
    training graph (it stays in the inference graph). Repeated facts are dropped.
    """
    url = f"{_OOKB_URL}/both-{name[5:]}"

    def facts(split: str, labelled: bool = False) -> list[tuple[str, str, str]]:
        rows = _read(f"{url}/{split}", root / split)
        return list(
            dict.fromkeys((h, r, t) for h, r, t, *label in rows if not labelled or label == ["1"])
        )

    train, aux, dev, test = facts("train"), facts("aux"), facts("dev", True), facts("test", True)
    seen = sorted({e for h, _, t in train for e in (h, t)})
    withheld = sorted(
        {e for split in (aux, dev, test) for h, _, t in split for e in (h, t)} - set(seen)
    )
    entities = {e: i for i, e in enumerate(seen + withheld)}
    relations = {
        r: i
        for i, r in enumerate(sorted({r for split in (train, aux, dev, test) for _, r, _ in split}))
    }

    def tensor(split: list[tuple[str, str, str]]) -> Tensor:
        return torch.tensor([[entities[h], relations[r], entities[t]] for h, r, t in split])

    train_facts = tensor(train)
    perm = torch.randperm(len(train_facts), generator=torch.Generator().manual_seed(SPLIT_SEED))
    held_out = 2 * int(0.05 * len(perm))
    return Dataset(
        name,
        len(seen),
        len(entities),
        len(relations),
        train=train_facts[perm[held_out:]],
        inference=tensor(train + aux),
        valid=tensor(dev),
        test=tensor(test),
    )
