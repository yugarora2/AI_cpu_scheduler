"""
Look at the dataset BEFORE training anything.   python inspect_data.py
Checks whether the labels are interesting (a mix of winners) or trivial.
"""
import pandas as pd
from config import FINAL_TEST_SEED_START
from candidates import LABEL_NAMES as NAMES
from scoring import add_labels

df = pd.read_csv("data/dataset.csv")
df = add_labels(df[df.seed < FINAL_TEST_SEED_START])
print(f"{len(df)} development workloads\n")

print("== Best scheduler overall (share of workloads) ==")
share = df["label"].value_counts(normalize=True)
print((share * 100).round(1).astype(str).add("%").to_string())

print("\n== Best scheduler per family (% of workloads) ==")
print((pd.crosstab(df["family"], df["label"], normalize="index") * 100).round(0).astype(int).to_string())

print("\n== Mean score per scheduler (0 = always best, lower is better) ==")
means = {c: df[f"{c}__score"].mean() for c in NAMES}
for c, v in sorted(means.items(), key=lambda kv: kv[1]):
    print(f"  {c:12s} {v:.3f}")

tied = (df["n_tied_best"] > 1).mean()
print(f"\nWorkloads where 2+ schedulers tie for best: {tied * 100:.0f}%")

top = share.iloc[0]
print("\n== Verdict ==")
if top > 0.60:
    print(f"'{share.index[0]}' wins {top * 100:.0f}% of the time -> too easy. "
          f"Adjust WEIGHTS in config.py (more switches/max_wt) or add workload variety.")
else:
    print("Labels are reasonably spread out. Good enough to train on.")
