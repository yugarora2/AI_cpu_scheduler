"""
Run:  python test_schedulers.py
Checks the algorithms against textbook answers and against properties that
must always hold. Run this BEFORE generating any data.
"""
from schedulers import fcfs, sjf, round_robin
from candidates import CANDIDATES
from workloads import make_workload
from config import FAMILIES


def P(pid, at, bt, pr=0):
    return {"pid": pid, "at": at, "bt": bt, "priority": pr}


def test_fcfs_textbook():
    ps = [P("P1", 0, 24), P("P2", 0, 3), P("P3", 0, 3)]
    assert abs(fcfs(ps)["avg_wt"] - 17.0) < 1e-9


def test_rr_textbook():
    ps = [P("P1", 0, 24), P("P2", 0, 3), P("P3", 0, 3)]
    assert abs(round_robin(ps, 4)["avg_wt"] - 17 / 3) < 1e-9


def test_sjf_textbook():
    ps = [P("P1", 0, 6), P("P2", 0, 8), P("P3", 0, 7), P("P4", 0, 3)]
    assert abs(sjf(ps)["avg_wt"] - 7.0) < 1e-9


def test_fcfs_idle_gap():                      # regression test for the old idle-time bug
    assert fcfs([P("P1", 5, 3)])["avg_rt"] == 0


def test_input_not_modified():
    ps = make_workload("mixed", 1)
    before = [dict(p) for p in ps]
    for fn in CANDIDATES.values():
        fn(ps)
    assert ps == before


def test_invariants_on_random_workloads():
    for family in FAMILIES:
        for seed in range(15):
            ps = make_workload(family, seed)
            res = {name: fn(ps) for name, fn in CANDIDATES.items()}
            # every scheduler here is work-conserving => identical finishing time
            assert len({r["makespan"] for r in res.values()}) == 1, (family, seed)
            # SRTF is provably optimal for average waiting time
            best_wt = min(r["avg_wt"] for r in res.values())
            assert abs(res["SRTF"]["avg_wt"] - best_wt) < 1e-9, (family, seed)
            # waiting and turnaround differ only by the (fixed) average burst
            for r in res.values():
                avg_bt = sum(p["bt"] for p in ps) / len(ps)
                assert abs((r["avg_tat"] - r["avg_wt"]) - avg_bt) < 1e-9


def test_chart_and_per_process_returned():
    ps = [P("P1", 0, 7, 3), P("P2", 1, 6, 2), P("P3", 2, 5, 1), P("P4", 5, 3, 2), P("P5", 8, 4, 1)]
    from schedulers import priority_p
    r = priority_p(ps)
    assert [(c[0], c[1], c[2]) for c in r["chart"]][:3] == [("P1", 0, 1), ("P2", 1, 2), ("P3", 2, 7)]
    assert abs(r["avg_wt"] - 7.6) < 1e-9 and r["per_process"]["P1"]["wt"] == 18


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
