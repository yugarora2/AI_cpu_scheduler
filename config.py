"""
Shared settings. EVERYONE on the team must use the same version of this file.

If you change anything in schedulers.py, workloads.py, features.py or the
candidate list, bump SIM_VERSION so that data made with different code is
never mixed together by accident (merge_data.py refuses mixed versions).
"""

SIM_VERSION = 1

# Workload families (see workloads.py for what each one means)
FAMILIES = ["cpu_bound", "interactive", "mixed", "heavy_tail", "bursty", "batch"]

# Definition of "best scheduler" for one workload.
# Each metric is normalised to 0..1 ACROSS the candidates (0 = best candidate,
# 1 = worst candidate) and then combined with these weights. Lowest score wins.
# NOTE: avg turnaround and throughput are deliberately NOT here: turnaround =
# waiting + burst (burst average is the same for every scheduler), and all our
# schedulers are work-conserving so throughput is identical for all of them.
WEIGHTS = {
    "avg_wt": 0.45,      # average waiting time
    "avg_rt": 0.25,      # average response time
    "max_wt": 0.15,      # worst waiting time (starvation / fairness)
    "switches": 0.15,    # number of context switches (cost of preemption)
}

# Raw metrics saved for every (workload, candidate) pair
METRIC_KEYS = ["avg_wt", "avg_tat", "avg_rt", "max_wt", "switches", "makespan", "throughput"]

# Seeds >= this value are RESERVED for the final test set.
# Nobody trains or tunes on them. They are only used once, at the very end.
FINAL_TEST_SEED_START = 900_000

# SJF and SRTF need to know each job's TRUE burst time in advance. A real OS cannot
# know that, and in our simulator they would win ~90% of workloads (SRTF is
# provably optimal for waiting time), which would make the ML task trivial.
# So by default they are still simulated and saved (as reference/oracle lines in
# the results) but they are NOT allowed to be the "best scheduler" label.
# Set the flag to True to let them compete as well (the dataset will be much easier).
CLAIRVOYANT = ["SJF", "SRTF"]
LABEL_INCLUDE_CLAIRVOYANT = False
