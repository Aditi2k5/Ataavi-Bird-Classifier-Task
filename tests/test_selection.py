import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from download_data import select_mix

SEL = {"min_overlap_frac": 0.25, "min_lowgrade_frac": 0.35,
       "low_grades": ["C", "D", "E", ""]}


def _pool(n=200):
    pool = []
    for i in range(n):
        q = "A" if i < 120 else ("B" if i < 160 else ("C" if i < 185 else "D"))
        also = ["Some bird"] if i % 10 == 0 else []      # ~20 overlaps
        pool.append({"id": str(i), "q": q, "also": also})
    return pool


def test_quotas_met_when_available():
    chosen = select_mix(_pool(), cap=70, sel=SEL)
    assert len(chosen) == 70
    n_ov = sum(bool(r["also"]) for r in chosen)
    n_lg = sum(r["q"] in {"C", "D", "E", ""} for r in chosen)
    assert n_ov >= min(20, round(0.25 * 70)) - 0   # capped by the 20 available
    assert n_lg >= round(0.35 * 70)


def test_rare_species_takes_all():
    rare = [{"id": str(i), "q": "C", "also": []} for i in range(12)]
    assert len(select_mix(rare, cap=999, sel=SEL)) == 12


if __name__ == "__main__":
    test_quotas_met_when_available()
    test_rare_species_takes_all()
    print("test_selection: OK")
