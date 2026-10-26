"""
Generate dataset rows. THIS is the script each teammate runs.

Each row = one workload: its features + the raw metrics of EVERY candidate scheduler.

Example (person 2 on the team):
    python generate_data.py --seed-start 10000 --per-family 500 --out data/part_person2.csv

Every person must use a DIFFERENT --seed-start (see README for the table), so
nobody generates the same workloads twice. Same seed => same workload, always.
"""
import argparse
import csv
import time
from multiprocessing import Pool, cpu_count

from config import FAMILIES, SIM_VERSION, METRIC_KEYS
from candidates import CANDIDATES
from workloads import make_workload
from features import extract_features, FEATURE_COLS


def simulate_one(job):
    family, seed = job
    ps = make_workload(family, seed)
    row = {"sim_version": SIM_VERSION, "family": family, "seed": seed}
    row.update(extract_features(ps))
    for name, fn in CANDIDATES.items():
        metrics = fn(ps)
        for k in METRIC_KEYS:
            row[f"{name}__{k}"] = metrics[k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True, help="first seed of YOUR range")
    ap.add_argument("--per-family", type=int, default=500, help="workloads per family")
    ap.add_argument("--out", required=True, help="output csv, e.g. data/part_alice.csv")
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    jobs = [(f, args.seed_start + i) for f in args.families for i in range(args.per_family)]
    print(f"Generating {len(jobs)} workloads x {len(CANDIDATES)} schedulers "
          f"with {args.workers} worker(s)...")

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(simulate_one, jobs, chunksize=10), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} done  ({el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left)")

    rows.sort(key=lambda r: (r["family"], r["seed"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":      # this guard is required for multiprocessing on Windows
    main()
