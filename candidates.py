"""
The list of 'candidate schedulers' the ML model chooses between.
Order matters: it is simplest-first, and ties in the score go to the earlier one.
"""
from config import CLAIRVOYANT, LABEL_INCLUDE_CLAIRVOYANT
from schedulers import fcfs, sjf, srtf, priority_np, priority_p, round_robin, mlfq

CANDIDATES = {
    "FCFS": lambda p: fcfs(p),
    "SJF": lambda p: sjf(p),
    "SRTF": lambda p: srtf(p),
    "PRIO_NP": lambda p: priority_np(p),
    "PRIO_P": lambda p: priority_p(p),
    # q=q binds the current value of q (otherwise every lambda would use 16)
    **{f"RR_q{q}": (lambda p, q=q: round_robin(p, q)) for q in (1, 2, 4, 8, 16)},
    "MLFQ_2_4_8": lambda p: mlfq(p, (2, 4, 8)),
    "MLFQ_4_8_16": lambda p: mlfq(p, (4, 8, 16)),
}

NAMES = list(CANDIDATES)

# Schedulers that are allowed to win (the ML model chooses among these)
LABEL_NAMES = [n for n in NAMES if LABEL_INCLUDE_CLAIRVOYANT or n not in CLAIRVOYANT]
