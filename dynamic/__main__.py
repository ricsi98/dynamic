"""Train DyNAMiC on one split and report its unseen-entity metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

from .data import DATASETS, load_dataset
from .evaluate import evaluate
from .model import ModelConfig
from .train import TrainConfig, train

TEST_CANDIDATES = 500  # negatives per direction in the reported ranking


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dynamic", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="raw data cache")
    parser.add_argument("--out", type=Path, help="write metrics.json and model.pt here")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for config in (ModelConfig, TrainConfig):
        group = parser.add_argument_group(config.__name__)
        for f in fields(config):
            flag = "--" + f.name.replace("_", "-")
            if isinstance(f.default, bool):
                kind = {"action": argparse.BooleanOptionalAction}
            else:
                kind = {"type": type(f.default), "metavar": type(f.default).__name__.upper()}
            group.add_argument(flag, default=f.default, help=f"default: {f.default}", **kind)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # epoch logs show up live in log files
    model_config = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig)})
    train_config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})

    dataset = load_dataset(args.dataset, args.data_dir)
    print(
        f"{dataset.name}: {dataset.num_seen:,} seen / {dataset.num_entities:,} entities, "
        f"{len(dataset.train):,} training facts, {len(dataset.valid):,} valid / "
        f"{len(dataset.test):,} test queries"
    )
    model, history = train(dataset, model_config, train_config, args.device)
    results = {
        split: evaluate(model, dataset, split, TEST_CANDIDATES) for split in ("valid", "test")
    }
    for split, m in results.items():
        print(
            f"{split:5s}  MRR {m['mrr']:.4f}  H@1 {m['hits@1']:.4f}  H@3 {m['hits@3']:.4f}  "
            f"H@10 {m['hits@10']:.4f}"
        )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        best = max(history, key=lambda e: e["valid"]["mrr"])
        (args.out / "metrics.json").write_text(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "model": asdict(model_config),
                    "training": asdict(train_config),
                    "best_epoch": best["epoch"],
                    **results,
                    "history": history,
                },
                indent=2,
            )
        )
        torch.save(model.state_dict(), args.out / "model.pt")


if __name__ == "__main__":
    main()
