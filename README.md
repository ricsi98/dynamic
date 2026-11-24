# DyNAMiC

Code for *DyNAMiC: Limits and Design Choices of One-Hop Encoders for Inductive Knowledge
Graph Completion* (Gábor Szűcs and Richárd Kiss).

DyNAMiC scores a candidate triple (h, r, t) without an embedding of h or t. Each endpoint
is represented by K sampled facts (neighbour, relation) around it; the two fact sets
attend to each other through differential attention, each is mean-pooled, and an MLP
scores the pair together with the query relation. An entity withheld from training can
therefore be scored as soon as it has facts in the graph, and a seen neighbour
contributes its trained embedding.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python -m dynamic --dataset WN18RR-semi --seed 42 --out runs/WN18RR-semi/dynamic/42
```

This downloads the raw files into `data/`, trains with the paper's configuration, keeps
the epoch with the best validation MRR, and prints MRR and Hits@{1,3,10} on the
validation and test queries. With `--out` it also writes `metrics.json` (configuration,
per-epoch log, results) and `model.pt`. Every hyperparameter is a flag (see
`uv run python -m dynamic --help`); `--relation-only` trains the relation-only variant.

## Datasets

| Split | seen entities | all entities | relations | training facts | inference facts | valid | test |
|---|---|---|---|---|---|---|---|
| `FB15k-237-semi` | 13,087 | 14,541 | 237 | 223,101 | 272,115 | 3,164 | 3,663 |
| `WN18RR-semi` | 36,849 | 40,943 | 11 | 70,366 | 86,835 | 560 | 579 |
| `OOKB-1000` | 36,237 | 38,165 | 11 | 82,235 | 109,788 | 1,768 | 476 |
| `OOKB-3000` | 32,118 | 37,978 | 11 | 62,515 | 107,214 | 1,198 | 1,349 |
| `OOKB-5000` | 28,601 | 37,615 | 11 | 50,640 | 103,917 | 897 | 2,124 |

Every split has a training graph over the seen entities and an inference graph that adds
the withheld entities and their facts.