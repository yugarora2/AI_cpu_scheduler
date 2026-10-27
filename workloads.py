"""
Synthetic workload generator.

A workload = a list of processes {"pid", "at", "bt", "priority"} with INTEGER times.
Each workload comes from one of several 'families' (different shapes of traffic).
The same (family, seed) always produces exactly the same workload.

Per workload we also randomise:
  * the number of processes (10..60)
  * a size scale for the bursts
  * the LOAD (rho): total work / time over which it arrives. rho < 1 means the
    CPU is mostly idle (all schedulers behave alike), rho > 1 means a queue
    builds up (the choice of scheduler matters a lot).
  * how priorities are assigned (random, or favouring short jobs)
"""
import numpy as np
from config import FAMILIES

MAX_BURST = 150


def _bursts(family, n, rng):
    if family == "cpu_bound":                       # few long jobs
        return rng.uniform(20, 60, n)
    if family == "interactive":                     # many short jobs
        return rng.exponential(3, n) + 1
    if family == "mixed":                           # 80% short, 20% long
        is_long = rng.random(n) < 0.2
        return np.where(is_long, rng.uniform(30, 70, n), rng.exponential(4, n) + 1)
    if family == "heavy_tail":                      # mostly tiny, a few huge (Pareto)
        return (rng.pareto(1.5, n) + 1) * 3
    if family == "bursty":                          # medium jobs
        return rng.exponential(8, n) + 1
    if family == "batch":                           # classic textbook: everything at once
        return rng.uniform(1, 40, n)
    raise ValueError(f"unknown family {family}")


def make_workload(family, seed):
    rng = np.random.default_rng([FAMILIES.index(family), int(seed)])
    n = int(rng.integers(10, 61))

    bursts = _bursts(family, n, rng) * rng.uniform(0.7, 1.5)
    bursts = np.clip(np.round(bursts), 1, MAX_BURST).astype(int)

    if family == "batch":
        arrivals = np.zeros(n, dtype=int)
        if rng.random() < 0.5:                      # sometimes a few late arrivals
            late = rng.random(n) < 0.2
            arrivals[late] = rng.integers(1, 30, late.sum())
    else:
        rho = rng.uniform(0.4, 1.6)
        mean_gap = bursts.mean() / rho
        if family == "bursty":                      # clumps of arrivals, then long silences
            tiny = rng.random(n) < 0.8
            gaps = np.where(tiny, rng.exponential(0.05 * mean_gap, n),
                            rng.exponential(4.8 * mean_gap, n))
        else:                                       # steady random arrivals
            gaps = rng.exponential(mean_gap, n)
        arrivals = np.cumsum(np.round(gaps)).astype(int)
        arrivals -= arrivals.min()

    prio = rng.integers(0, 5, n)
    return [
        {"pid": f"P{i}", "at": int(arrivals[i]), "bt": int(bursts[i]), "priority": int(prio[i])}
        for i in range(n)
    ]
