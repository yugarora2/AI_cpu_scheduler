"""
Run:  python test_recommend.py
Tests the app logic (input checking, ranking, shortlist, winner) with a tiny model built on the fly (about 10 seconds).
"""
import pandas as pd
from config import FAMILIES
from generate_data import simulate_one
from modeling import fit_rankers
from scoring import add_labels
from candidates import LABEL_NAMES
from features import FEATURE_COLS
from recommend import validate, recommend, score_set, load_bundle
import joblib, io


def frame(rows):
    return pd.DataFrame(rows, columns=["pid", "at", "bt", "priority"])


HOSPITAL = [("P1", 0, 7, 3), ("P2", 1, 6, 2), ("P3", 2, 5, 1), ("P4", 5, 3, 2), ("P5", 8, 4, 1)]


def tiny_bundle():
    rows = [simulate_one((f, s)) for f in FAMILIES for s in range(25)]
    df = add_labels(pd.DataFrame(rows))
    models = fit_rankers(df, n_jobs=1, n_estimators=20)
    return {"models": models, "cands": LABEL_NAMES, "features": FEATURE_COLS, "meta": {}}


def test_validate_good_and_bad_input():
    procs, errs, warns = validate(frame(HOSPITAL))
    assert not errs and len(procs) == 5 and warns            # warns: fewer than 10 processes
    assert validate(frame([("P1", 0, 5, 0), ("P1", 1, 5, 0), ("P2", 2, 3, 0)]))[1]       # duplicate names
    assert validate(frame([("P1", -1, 5, 0), ("P2", 1, 5, 0), ("P3", 2, 3, 0)]))[1]      # negative arrival
    assert validate(frame([("P1", 0, 0, 0), ("P2", 1, 5, 0), ("P3", 2, 3, 0)]))[1]       # zero burst
    assert validate(frame([("P1", 0, 2.5, 0), ("P2", 1, 5, 0), ("P3", 2, 3, 0)]))[1]     # not a whole number
    assert validate(frame([("P1", 0, 5, 0), ("P2", 1, 5, 0)]))[1]                         # fewer than 3 processes
    procs, errs, _ = validate(frame([("P1", 0, 5, None), ("P2", 1, 5, None), ("P3", 2, 3, None)]))
    assert not errs and all(p["priority"] == 0 for p in procs)                            # blank priority -> 0


def test_recommend_end_to_end():
    bundle = tiny_bundle()
    procs, _, _ = validate(frame(HOSPITAL))
    before = [dict(p) for p in procs]
    out = recommend(bundle, procs, k=3, method="classifier", verify=True)
    assert procs == before                                   # input not modified
    assert len(out["shortlist"]) == 3 and out["winner"] in out["shortlist"]
    sc = out["scores"]
    assert -1e-12 <= min(sc.values()) and max(sc.values()) <= 1 + 1e-12     # weights sum to 1, each metric scaled 0..1
    assert set(out["verify"]["all_results"]) >= set(LABEL_NAMES) and "SRTF" in out["verify"]["all_results"]
    # the winner is the best of the shortlist under the stated score
    assert sc[out["winner"]] == min(sc.values())
    # regressor ranking and k=1 also work
    assert recommend(bundle, procs, k=1, method="regressor", verify=False)["shortlist"]


def test_hospital_question_matches_hand_calculation():
    bundle = tiny_bundle()
    procs, _, _ = validate(frame(HOSPITAL))
    out = recommend(bundle, procs, k=10, method="classifier", verify=False)   # k=10 simulates all candidates
    r = out["results"]["PRIO_P"]
    assert abs(r["avg_wt"] - 7.6) < 1e-9                                      # worked out by hand: 38 / 5
    assert [r["per_process"][p]["wt"] for p in ("P1", "P2", "P3", "P4", "P5")] == [18, 9, 0, 11, 0]


def test_model_file_roundtrip():
    bundle = tiny_bundle()
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    buf.seek(0)
    assert load_bundle(buf)["cands"] == LABEL_NAMES


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
