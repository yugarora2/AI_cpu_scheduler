"""
Merge everyone's part files into data/dataset.csv.

    python merge_data.py                       # merges data/part_*.csv
    python merge_data.py data/a.csv data/b.csv # or list files explicitly

Refuses to mix data made with different SIM_VERSIONs, and drops duplicate
(family, seed) rows (which would mean two people used overlapping seeds).
"""
import glob
import sys
import pandas as pd
from config import SIM_VERSION

files = sys.argv[1:] or sorted(glob.glob("data/part_*.csv"))
if not files:
    sys.exit("No files found. Put part files in data/ named part_*.csv")

parts = []
for f in files:
    d = pd.read_csv(f)
    versions = set(d["sim_version"])
    print(f"{f}: {len(d)} rows, sim_version={sorted(versions)}")
    if versions != {SIM_VERSION}:
        sys.exit(f"ERROR: {f} was made with sim_version {versions}, but config.py says {SIM_VERSION}. "
                 f"Regenerate it with the current code.")
    parts.append(d)

cols = set(parts[0].columns)
for f, d in zip(files, parts):
    if set(d.columns) != cols:
        sys.exit(f"ERROR: {f} has different columns (different code version?).")

df = pd.concat(parts, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["family", "seed"]).reset_index(drop=True)
if len(df) < before:
    print(f"WARNING: dropped {before - len(df)} duplicate rows (overlapping seed ranges?)")

df.to_csv("data/dataset.csv", index=False)
print(f"\nSaved data/dataset.csv with {len(df)} rows")
print(df["family"].value_counts().to_string())
