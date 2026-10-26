"""Turn raw metrics into a score per candidate and a 'best scheduler' label."""
import numpy as np
from config import WEIGHTS
from candidates import LABEL_NAMES as NAMES


def add_labels(df, weights=None):
    weights = weights or WEIGHTS
    total = np.zeros((len(df), len(NAMES)))
    for metric, w in weights.items():
        M = df[[f"{c}__{metric}" for c in NAMES]].to_numpy(float)
        lo = M.min(axis=1, keepdims=True)
        hi = M.max(axis=1, keepdims=True)
        span = hi - lo
        norm = np.divide(M - lo, span, out=np.zeros_like(M), where=span > 0)   # 0 best .. 1 worst
        total += w * norm

    out = df.copy()
    for j, c in enumerate(NAMES):
        out[f"{c}__score"] = total[:, j]
    best = total.min(axis=1)
    out["label"] = [NAMES[j] for j in total.argmin(axis=1)]          # ties -> earliest candidate
    out["n_tied_best"] = (total <= best[:, None] + 1e-12).sum(axis=1)
    return out
