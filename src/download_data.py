from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, load_api_key, ensure_dir
from xc_client import XenoCanto, build_query, is_overlap


def dedupe(recs):
    seen, out = set(), []
    for r in recs:
        rid = str(r.get("id"))
        if rid and rid not in seen:
            seen.add(rid); out.append(r)
    return out


def select_mix(recs, cap, sel):
    recs = dedupe(recs)
    if len(recs) <= cap:
        return recs
    low = set(g if g is not None else "" for g in sel["low_grades"])
    by_id = {str(r["id"]): r for r in recs}
    def lowgrade(r):
        return (r.get("q") or "") in low
    overlap_ids = [i for i, r in by_id.items() if is_overlap(r)]
    lowgrade_ids = [i for i, r in by_id.items() if lowgrade(r)]
    want_ov = min(len(overlap_ids), math.ceil(sel["min_overlap_frac"] * cap))
    want_lg = min(len(lowgrade_ids), math.ceil(sel["min_lowgrade_frac"] * cap))
    chosen = []
    def add(ids, n):
        for i in ids:
            if len(chosen) >= cap or n <= 0:
                break
            if i not in chosen:
                chosen.append(i); n -= 1
        return n
    add(overlap_ids, want_ov)
    still_lg = want_lg - sum(1 for i in chosen if lowgrade(by_id[i]))
    add([i for i in lowgrade_ids if i not in chosen], still_lg)
    add(list(by_id.keys()), cap - len(chosen))
    return [by_id[i] for i in chosen[:cap]]


def manifest_row(sp, r, sel):
    low = set(g if g is not None else "" for g in sel["low_grades"])
    also = r.get("also") or []
    return {
        "label": sp["label"], "english_name": sp["en"], "tier": sp["tier"],
        "scientific": f"{r.get('gen','')} {r.get('sp','')}".strip(),
        "xc_id": r["id"], "url": f"https://xeno-canto.org/{r['id']}",
        "quality": r.get("q", ""), "length": r.get("length", ""),
        "type": r.get("type", ""), "country": r.get("cnt", ""),
        "recordist": r.get("rec", ""), "license": r.get("lic", ""),
        "also_species": "; ".join(str(a) for a in also if str(a).strip()),
        "has_overlap": is_overlap(r), "is_lowgrade": (r.get("q") or "") in low,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    xc = XenoCanto(load_api_key(), cfg["query"]["per_page"],
                   cfg["download"]["sleep_between_requests"])
    audio_dir = ensure_dir(cfg["paths"]["audio_dir"])
    rows = []

    for sp in cfg["species"]:
        query = build_query(sp, cfg["query"])
        print(f"\n=== {sp['label']} ({sp['en']}, tier={sp['tier']})\n    query: {query}")
        recs = xc.search(query)
        print(f"    found {len(recs)} on Xeno-canto")
        target = sp["max"]
        pool = select_mix(recs, min(len(recs), max(target * 4, target)), cfg["selection"])
        n_ov = sum(is_overlap(r) for r in pool)
        n_lg = sum((r.get("q") or "") in set(cfg["selection"]["low_grades"]) for r in pool)
        print(f"    candidate pool {len(pool)} (target {target}; {n_ov} overlap, {n_lg} low-grade in pool)")
        ok = 0
        for r in pool:
            if ok >= target:
                break
            dest = audio_dir / sp["label"] / f"{r['id']}.mp3"
            if args.dry_run:
                ok += 1
            elif (cfg["download"]["skip_existing"] and dest.exists()) or xc.download(r, dest):
                ok += 1
            else:
                continue
            rows.append(manifest_row(sp, r, cfg["selection"]))
        print(f"    {'listed' if args.dry_run else 'downloaded'} {ok} / {target}")
        if not args.dry_run and ok < target:
            print(f"    NOTE: only {ok} usable (rest were no-download/HTML at source)")

    man = Path(cfg["paths"]["manifest"])
    ensure_dir(man.parent)
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== recordings per class ===")
    counts = Counter(r["label"] for r in rows)
    for sp in cfg["species"]:
        print(f"  {sp['label']:24s} {counts.get(sp['label'],0):4d}   (tier: {sp['tier']})")
    print(f"\ntotal: {len(rows)}  overlap: {sum(r['has_overlap'] for r in rows)}  "
          f"low-grade: {sum(r['is_lowgrade'] for r in rows)}")
    print(f"[wrote] {man}")


if __name__ == "__main__":
    main()
