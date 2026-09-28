from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


def recording_level_split(
    recording_ids: Sequence[str],
    labels: Sequence[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 1337,
) -> Dict[str, set]:
    """Return {"train","val","test"} sets of unique recording ids.

    recording_ids / labels are parallel sequences at the *window* level (many rows
    per recording is fine — duplicates are collapsed). Stratification is per label.
    """
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = {}
    seen = set()
    for rid, lab in zip(recording_ids, labels):
        rid = str(rid)
        if rid in seen:
            continue
        seen.add(rid)
        by_label.setdefault(lab, []).append(rid)

    train, val, test = set(), set(), set()
    _, r_va, r_te = ratios
    for lab, rids in by_label.items():
        rids = list(rids)
        rng.shuffle(rids)
        n = len(rids)
        if n == 0:
            continue
        if n == 1:                      # too few to hold out
            train.add(rids[0]); continue
        if n == 2:
            train.add(rids[0]); test.add(rids[1]); continue
        n_te = max(1, round(n * r_te))
        n_va = max(1, round(n * r_va))
        n_te = min(n_te, n - 2)         # always leave >=2 for train
        n_va = min(n_va, n - 1 - n_te)
        test.update(rids[:n_te])
        val.update(rids[n_te:n_te + n_va])
        train.update(rids[n_te + n_va:])
    return {"train": train, "val": val, "test": test}


def assert_no_leakage(splits: Dict[str, set]) -> None:
    """Raise if any recording appears in more than one split."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert not (tr & va), "train/val overlap"
    assert not (tr & te), "train/test overlap"
    assert not (va & te), "val/test overlap"
