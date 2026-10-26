"""Turn a workload into a short list of numbers (the ML 'features')."""
import numpy as np

FEATURE_COLS = [
    "n", "burst_mean", "burst_cv", "burst_tail", "frac_short",
    "load", "gap_cv", "frac_simultaneous", "prio_std", "prio_burst_corr",
]


def extract_features(ps):
    b = np.array([p["bt"] for p in ps], float)
    a = np.sort([p["at"] for p in ps]).astype(float)
    pr = np.array([p["priority"] for p in ps], float)
    gaps = np.diff(a)
    span = max(a[-1] - a[0], 1.0)
    corr = np.corrcoef(pr, b)[0, 1] if pr.std() > 0 and b.std() > 0 else 0.0
    return {
        "n": len(ps),
        "burst_mean": b.mean(),
        "burst_cv": b.std() / b.mean(),                         # variability of job lengths
        "burst_tail": np.percentile(b, 90) / np.median(b),      # how heavy the tail is
        "frac_short": float((b < 0.5 * b.mean()).mean()),
        "load": b.sum() / span,                                 # work arriving per unit time
        "gap_cv": gaps.std() / (gaps.mean() + 1e-9),            # burstiness of arrivals
        "frac_simultaneous": float((gaps == 0).mean()),         # share of jobs arriving together
        "prio_std": pr.std(),
        "prio_burst_corr": corr,                                # does priority say anything about length?
    }
