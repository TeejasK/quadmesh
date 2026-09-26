"""
Human review of the risk_tier_classifier / abuse_pattern candidates.

There is no ground truth for these two roles anywhere. role_gen_data writes CANDIDATES (label_source=taxonomy_v1,
reviewed=false); a model trained on unreviewed guesses would only learn to repeat the guesser. Training and the
runtime treat a role as trained only when >= 500 rows are reviewed=true.

    python -m quadmesh.pipeline.review_queue review_queue.jsonl            # interactive
    python -m quadmesh.pipeline.review_queue review_queue.jsonl --stats

Keys:  a = approve as shown   e = edit the label   r = reject (deleted)   s = skip   q = save and quit
Approve only what YOU would sign. Add your own real prompts to the file (reviewed=true) - real requests
beat generated ones.
"""
import argparse
import json
import sys

from quadmesh.roles_io import RISK_TIERS, ABUSE_CATEGORIES


def _edit(row):
    if row["role"] == "risk_tier_classifier":
        t = input(f"  tier {RISK_TIERS}: ").strip()
        if t not in RISK_TIERS:
            return False
        row["answer"] = f"TIER: {t}\nWHY: " + (input("  why: ").strip() or "reviewed by a human")
    else:
        c = input(f"  category {ABUSE_CATEGORIES}: ").strip()
        if c not in ABUSE_CATEGORIES:
            return False
        row["answer"] = (f"ABUSE: {'no' if c == 'none' else 'yes'}\nCATEGORY: {c}\nWHY: "
                         + (input("  why: ").strip() or "reviewed by a human"))
    row["label_source"] = "human_edit"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    if a.stats:
        for role in ("risk_tier_classifier", "abuse_pattern"):
            rr = [r for r in rows if r["role"] == role]
            print(f"{role}: {sum(1 for r in rr if r['reviewed'])} reviewed / {len(rr)} rows (need >= 500 reviewed)")
        return
    keep = []
    try:
        for i, r in enumerate(rows):
            if r["reviewed"]:
                keep.append(r); continue
            print(f"\n[{i + 1}/{len(rows)}] {r['role']}\n  request: {r['prompt']}\n  proposed: {r['answer'].replace(chr(10), ' | ')}")
            k = input("  [a]pprove [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
            if k == "q":
                keep += rows[i:]; break
            if k == "r":
                continue
            if k == "a" or (k == "e" and _edit(r)):
                r["reviewed"] = True
            keep.append(r)
    finally:
        with open(a.path, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"saved {a.path}")


if __name__ == "__main__":
    main()
