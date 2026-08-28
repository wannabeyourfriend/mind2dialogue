"""Assemble the privileged-vs-blind comparison table from multibench results.

Reads evaluations/results/<arm>/<Bench>/ for arms base|privileged|blind and
prints a markdown table plus a JSON dump.

Usage (on the server, after run_ablation_evals.sh):
  python scripts/collect_ablation_results.py --results-root evaluations/results
"""

import argparse
import json
import re
from pathlib import Path

ARMS = ["base", "privileged", "blind"]


def find_json(root: Path, patterns):
    hits = []
    for pat in patterns:
        hits += sorted(root.rglob(pat))
    return hits


def read_bigtom(arm_dir: Path):
    out = {}
    for f in find_json(arm_dir, ["summary_*.json", "results_*/**/summary*.json"]):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for k, v in d.items():
            if isinstance(v, (int, float)) and re.search(r"acc|score", k, re.I):
                out[f"{f.parent.name}:{k}"] = v
    return out


def read_prefeval(arm_dir: Path):
    out = {}
    for f in find_json(arm_dir, ["*_summary.json", "**/*summary*.json"]):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for k, v in d.items():
            if isinstance(v, (int, float)) and re.search(r"acc|rate|score|error", k, re.I):
                out[k] = v
    return out


def read_personamem(arm_dir: Path):
    out = {}
    for f in sorted(arm_dir.rglob("*summary*.txt")):
        txt = f.read_text()
        m = re.search(r"(?:overall\s+)?accuracy[^0-9]*([0-9.]+)", txt, re.I)
        if m:
            out[f.stem] = float(m.group(1))
    for f in sorted(arm_dir.rglob("evaluation_results_*.csv")):
        rows = f.read_text().splitlines()
        if len(rows) > 1:
            hdr = rows[0].split(",")
            if "is_correct" in hdr:
                idx = hdr.index("is_correct")
                vals = [r.split(",")[idx].strip().lower() for r in rows[1:] if r.strip()]
                tru = sum(1 for v in vals if v in ("true", "1"))
                if vals:
                    out[f.stem] = round(100 * tru / len(vals), 2)
    return out


READERS = {"BigTom": read_bigtom, "PrefEval": read_prefeval, "PersonaMem": read_personamem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="evaluations/results")
    args = ap.parse_args()
    root = Path(args.results_root)

    table = {}
    for arm in ARMS:
        arm_dir = root / arm
        table[arm] = {}
        if not arm_dir.exists():
            continue
        for bench, reader in READERS.items():
            for d in arm_dir.iterdir() if arm_dir.is_dir() else []:
                if d.is_dir() and bench.lower() in d.name.lower():
                    table[arm][bench] = reader(d)

    print(json.dumps(table, indent=2))
    metrics = sorted({f"{b}/{k}" for arm in table.values() for b, kv in arm.items() for k in kv})
    if metrics:
        print("\n| metric | " + " | ".join(ARMS) + " | blind−priv |")
        print("|---|" + "---|" * (len(ARMS) + 1))
        for m in metrics:
            b, k = m.split("/", 1)
            vals = [table.get(a, {}).get(b, {}).get(k) for a in ARMS]
            cells = ["—" if v is None else f"{v:.2f}" for v in vals]
            delta = (
                f"{vals[2] - vals[1]:+.2f}"
                if vals[1] is not None and vals[2] is not None
                else "—"
            )
            print(f"| {m} | " + " | ".join(cells) + f" | {delta} |")


if __name__ == "__main__":
    main()
