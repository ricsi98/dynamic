#!/usr/bin/env bash
# Train DyNAMiC with the paper's configuration on each split and seed, then print the
# mean and standard deviation over seeds. Pass dataset names to run a subset, e.g.
#   examples/run.sh WN18RR-semi OOKB-5000
set -euo pipefail

datasets=("$@")
((${#datasets[@]})) || datasets=(FB15k-237-semi WN18RR-semi OOKB-1000 OOKB-3000 OOKB-5000)
for dataset in "${datasets[@]}"; do
  for seed in 42 43 44; do
    uv run python -m dynamic --dataset "$dataset" --seed "$seed" --out "runs/$dataset/dynamic/$seed"
    # The relation-only variant, reported on OOKB.
    if [[ $dataset == OOKB-* ]]; then
      uv run python -m dynamic --dataset "$dataset" --seed "$seed" --relation-only \
        --out "runs/$dataset/relation-only/$seed"
    fi
  done
done
uv run python examples/summarize.py runs
