import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from splits import recording_level_split, assert_no_leakage


def _windows():
    """Fake window-level rows: many windows per recording, imbalanced classes."""
    rids, labs = [], []
    plan = {"cuckoo": 40, "magpie": 38, "bulbul": 42, "sunbird": 20,
            "cisticola": 18, "owl": 4}          # owl = rare
    rid = 0
    for lab, n_rec in plan.items():
        for _ in range(n_rec):
            rid += 1
            for _ in range(6):                  # 6 windows per recording
                rids.append(f"rec{rid}"); labs.append(lab)
    return rids, labs


def test_no_leakage():
    rids, labs = _windows()
    splits = recording_level_split(rids, labs, seed=1337)
    assert_no_leakage(splits)                    # raises if any overlap


def test_every_recording_assigned_once():
    rids, labs = _windows()
    splits = recording_level_split(rids, labs, seed=1337)
    all_ids = set(rids)
    union = splits["train"] | splits["val"] | splits["test"]
    assert union == all_ids


def test_rare_class_in_test():
    rids, labs = _windows()
    splits = recording_level_split(rids, labs, seed=1337)
    owl_recs = {r for r, l in zip(rids, labs) if l == "owl"}
    assert owl_recs & splits["test"], "rare class must appear in test"


if __name__ == "__main__":
    test_no_leakage()
    test_every_recording_assigned_once()
    test_rare_class_in_test()
    print("test_splits: OK")
