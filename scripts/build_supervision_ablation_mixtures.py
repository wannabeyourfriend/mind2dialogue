"""Build the matched privileged/blind SFT mixtures for the supervision ablation.

Emits two JSONL files in IDENTICAL order (same items, same shuffle, seed 42):
  training/data/train_privileged.jsonl — released targets (state-privileged teacher)
  training/data/train_blind.jsonl      — blind-teacher targets (no state access)

Pairs where the blind relabel is missing/unparseable are dropped from BOTH
arms so the two datasets stay item-for-item matched.

Composition per arm mirrors the paper's mix_v6 (n=8244) as closely as the
public release allows:
  dialogue 3312 = 1240 random_multisource (= paper's conv-v1realquery, exact)
                + 624 pilot-id conversations (= conv-v1pilot, sampled from 637)
                + 1448 deep-scenario conversations (= conv v4 5 slices,
                  stratified across affective/highfreq/lifelong by size)
  personamem_mcq 2052 (v2_rewritten, exact)
  prefeval_gen   1442 (sampled from 2065, = paper's qa-prefeval_gen)
  total 6806
The paper's 1438 qa-lamp_cls slice is absent from the public release and is
omitted from BOTH arms. bigtom_tom is released but never trained on (both arms).
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND_DIALOGUE = ROOT / "output/relabel_blind/dialogue_blind.jsonl"
BLIND_QA = ROOT / "output/relabel_blind/qa_blind.jsonl"
OUT_DIR = ROOT / "training/data"

sys.path.insert(0, str(ROOT / "scripts"))
from relabel_conversations_with_blind_teacher import load_conversations  # noqa: E402
from relabel_questions_with_blind_teacher import load_items  # noqa: E402

PILOT_RE = re.compile(r"^profile_\d+_scenario_\d+$")
N_REALQUERY, N_PILOT, N_DEEP, N_PREFEVAL = 1240, 624, 1448, 1442


def select_conversations(rows, rng):
    """Mirror the paper's 3312-conversation dialogue slice."""
    realquery, pilot, deep = [], [], []
    for r in rows:
        if r["family"] == "random_multisource":
            realquery.append(r)
        elif PILOT_RE.match(str(r["scenario_id"] or "")):
            pilot.append(r)
        else:
            deep.append(r)

    def take(pool, n):
        pool = sorted(pool, key=lambda r: r["hash"])
        if len(pool) <= n:
            return pool
        return rng.sample(pool, n)

    by_family = defaultdict(list)
    for r in deep:
        by_family[r["family"]].append(r)
    total_deep = sum(len(v) for v in by_family.values())
    deep_sel = []
    for fam in sorted(by_family):
        share = round(N_DEEP * len(by_family[fam]) / total_deep)
        deep_sel += take(by_family[fam], share)
    deep_sel = deep_sel[:N_DEEP]

    sel = take(realquery, N_REALQUERY) + take(pilot, N_PILOT) + deep_sel
    print(
        f"dialogue slice: realquery={min(len(realquery), N_REALQUERY)} "
        f"pilot={min(len(pilot), N_PILOT)} deep={len(deep_sel)} total={len(sel)}"
    )
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["privileged", "blind", "both"], default="both")
    args = ap.parse_args()
    need_blind = args.arm in ("blind", "both")

    blind_conv, blind_qa = {}, {}
    if BLIND_DIALOGUE.exists():
        for line in BLIND_DIALOGUE.read_text().splitlines():
            r = json.loads(line)
            blind_conv[r["content_hash"]] = r
    if BLIND_QA.exists():
        for line in BLIND_QA.read_text().splitlines():
            r = json.loads(line)
            blind_qa[r["sample_id"]] = r

    pairs = []  # (privileged_line, blind_line|None)
    drops = {"conv_missing": 0, "conv_empty": 0, "qa_missing": 0, "qa_unparsed": 0}
    rng = random.Random(42)

    # Selection is deterministic over the FULL released corpus so both arms
    # draw the identical slice regardless of when each is built.
    for row in select_conversations(load_conversations(), rng):
        rec = blind_conv.get(row["hash"])
        if need_blind and (rec is None or rec["empty_turns"]):
            drops["conv_missing" if rec is None else "conv_empty"] += 1
            continue
        msgs = row["messages"]
        blind_msgs = [dict(m) for m in msgs]
        n_asst = 0
        ok = True
        for i, m in enumerate(msgs[1:]):
            if m["role"] != "assistant":
                continue
            n_asst += 1
            t = rec["blind_turns"].get(str(i)) if rec else None
            if not t:
                ok = need_blind is False
                if need_blind:
                    break
                continue
            blind_msgs[1 + i]["content"] = t
        if not ok or n_asst == 0:
            drops["conv_empty"] += 1
            continue
        meta = {
            "source": "dialogue",
            "family": row["family"],
            "persona_id": row["persona_id"],
            "scenario_id": row["scenario_id"],
            "content_hash": row["hash"],
        }
        pairs.append(
            (
                {"messages": msgs, "metadata": {**meta, "arm": "privileged"}},
                {"messages": blind_msgs, "metadata": {**meta, "arm": "blind"}},
            )
        )

    qa_items = [i for i in load_items() if i["qa_style"] != "bigtom_tom"]
    qa_items = [
        i
        for i in qa_items
        if not (i["qa_style"] == "personamem_mcq" and i["rewrite_status"] != "v2_rewritten")
    ]
    prefeval = sorted(
        (i for i in qa_items if i["qa_style"] == "prefeval_gen"), key=lambda i: i["sample_id"]
    )
    if len(prefeval) > N_PREFEVAL:
        keep = set(i["sample_id"] for i in rng.sample(prefeval, N_PREFEVAL))
        qa_items = [
            i for i in qa_items if i["qa_style"] != "prefeval_gen" or i["sample_id"] in keep
        ]

    for item in qa_items:
        style = item["qa_style"]
        rec = blind_qa.get(item["sample_id"])
        if need_blind:
            if rec is None or not rec.get("blind_target"):
                drops["qa_missing"] += 1
                continue
            if style == "personamem_mcq" and not rec.get("blind_letter"):
                drops["qa_unparsed"] += 1
                continue
        msgs = item["messages"]
        blind_msgs = [dict(m) for m in msgs]
        if rec and rec.get("blind_target"):
            blind_msgs[-1]["content"] = rec["blind_target"]
        meta = {
            "source": "qa",
            "qa_style": style,
            "sample_id": item["sample_id"],
            "stored_letter": item["stored_letter"] or None,
            "blind_letter": rec.get("blind_letter"),
        }
        pairs.append(
            (
                {"messages": msgs, "metadata": {**meta, "arm": "privileged"}},
                {"messages": blind_msgs, "metadata": {**meta, "arm": "blind"}},
            )
        )

    random.Random(42).shuffle(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.arm in ("privileged", "both"):
        with open(OUT_DIR / "train_privileged.jsonl", "w", encoding="utf-8") as fp:
            for priv, _ in pairs:
                fp.write(json.dumps(priv, ensure_ascii=False) + "\n")
    if need_blind:
        with open(OUT_DIR / "train_blind.jsonl", "w", encoding="utf-8") as fb:
            for _, blind in pairs:
                fb.write(json.dumps(blind, ensure_ascii=False) + "\n")
    # manifest so the later-built arm reproduces this exact slice
    manifest = [
        p["metadata"].get("content_hash") or p["metadata"].get("sample_id") for p, _ in pairs
    ]
    (OUT_DIR / f"manifest_{args.arm}.json").write_text(json.dumps(manifest, indent=1))

    n_conv = sum(1 for p, _ in pairs if p["metadata"]["source"] == "dialogue")
    n_qa = len(pairs) - n_conv
    mcq = [p["metadata"] for p, _ in pairs if p["metadata"].get("qa_style") == "personamem_mcq"]
    agree = sum(1 for m in mcq if m["blind_letter"] and m["stored_letter"] and m["blind_letter"].lower() == m["stored_letter"].lower())
    print(f"pairs: {len(pairs)} (dialogue {n_conv}, qa {n_qa}) | drops: {drops}")
    if mcq:
        print(f"personamem_mcq blind-vs-stored letter agreement: {agree}/{len(mcq)} = {agree/len(mcq):.1%}")


if __name__ == "__main__":
    main()
