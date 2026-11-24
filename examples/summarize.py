"""Test MRR and Hits@10 over seeds (mean ± population std) of runs/<dataset>/<model>/<seed>."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

runs = defaultdict(list)
for path in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs").glob("*/*/*/metrics.json")):
    runs[path.parts[-4], path.parts[-3]].append(json.loads(path.read_text())["test"])

print(f"{'dataset':<16} {'model':<14} {'seeds':>5}  {'MRR':^13}  {'H@10':^13}")
for (dataset, model), tests in runs.items():
    cells = [
        f"{statistics.mean(v):.3f} ± {statistics.pstdev(v):.3f}"
        for v in ([t["mrr"] for t in tests], [t["hits@10"] for t in tests])
    ]
    print(f"{dataset:<16} {model:<14} {len(tests):>5}  {cells[0]}  {cells[1]}")
