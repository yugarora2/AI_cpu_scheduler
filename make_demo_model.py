"""
Build a SMALL demo model so you can try the app on your laptop right away.

    python make_demo_model.py

Uses one CPU core and takes under a minute. The model is trained on only a few hundred
workloads, so it is for testing the interface ONLY. For real results train on Colab
(colab/02_merge_and_train.ipynb) and copy the downloaded ranker.joblib into models/.
"""
import datetime
import os
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import FAMILIES
from candidates import CANDIDATES, LABEL_NAMES
from features import FEATURE_COLS, extract_features
from generate_data import simulate_one
from modeling import fit_rankers
from scoring import add_labels

PER_FAMILY = 60

if os.path.exists("models/ranker.joblib"):
    ans = input("models/ranker.joblib already exists. Overwrite it with a demo model? [y/N] ")
    if ans.strip().lower() != "y":
        raise SystemExit("Nothing changed.")

rows = [simulate_one((f, s)) for f in FAMILIES for s in range(PER_FAMILY)]
df = add_labels(pd.DataFrame(rows))
models = fit_rankers(df, n_jobs=1, n_estimators=60)

os.makedirs("models", exist_ok=True)
meta = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "python": platform.python_version(), "n_dev_workloads": len(df),
        "created": datetime.date.today().isoformat(), "demo": True}
joblib.dump({"models": models, "cands": LABEL_NAMES, "features": FEATURE_COLS, "meta": meta},
            "models/ranker.joblib", compress=3)
print(f"Saved models/ranker.joblib (DEMO model, {len(df)} workloads).")
