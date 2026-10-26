"""
CPU scheduling algorithms (headless version of the original code).

Every algorithm:
  * takes a list of process dicts: {"pid": str, "at": int, "bt": int, "priority": int}
    (lower priority number = higher priority; times must be integers)
  * does NOT modify its input
  * returns a dict of metrics (see finish())
  * prints the Gantt chart / table only when verbose=True
"""
from collections import deque


def add_segment(chart, pid, start, end):
    if start == end:
        return
    if chart and chart[-1][0] == pid and chart[-1][2] == start:
        chart[-1] = (pid, chart[-1][1], end)
    else:
        chart.append((pid, start, end))


def text_gantt(chart):
    """Compact text Gantt chart with the time boundaries lined up under the blocks."""
    cells = [f"| {pid} " for pid, _, _ in chart]
    starts = [str(seg[1]) for seg in chart]
    widths = [max(len(c), len(t) + 1) for c, t in zip(cells, starts)]
    line1 = "".join(c.ljust(w) for c, w in zip(cells, widths)) + "|"
    line2 = "".join(t.ljust(w) for t, w in zip(starts, widths)) + str(chart[-1][2])
    return line1 + "\n" + line2


def finish(processes, chart, verbose=False):
    # Collect the run segments of each process in ONE pass over the chart
    segs = {}
    for name, s, e in chart:
        if name != "IDLE":
            segs.setdefault(name, []).append((s, e))

    metrics = {}
    for p in processes:
        pid = p["pid"]
        segments = segs[pid]
        # sanity check: a process must run for exactly its burst time
        assert sum(e - s for s, e in segments) == p["bt"], f"{pid} ran the wrong amount"

        first_start = min(s for s, e in segments)
        ct = max(e for s, e in segments)
        tat = ct - p["at"]
        wt = tat - p["bt"]
        rt = first_start - p["at"]
        metrics[pid] = {"at": p["at"], "bt": p["bt"], "ct": ct, "tat": tat, "wt": wt, "rt": rt}

    n = len(processes)

    def avg(key):
        return sum(m[key] for m in metrics.values()) / n

    if verbose:
        print("\nGantt Chart:")
        print(text_gantt(chart))
        print("\nPID\tAT\tBT\tCT\tTAT\tWT\tRT")
        for p in processes:
            m = metrics[p["pid"]]
            print(f"{p['pid']}\t{m['at']}\t{m['bt']}\t{m['ct']}\t{m['tat']}\t{m['wt']}\t{m['rt']}")
        print(f"\nAverage Waiting Time = {avg('wt'):.2f}")
        print(f"Average Turnaround Time = {avg('tat'):.2f}")
        print(f"Average Response Time = {avg('rt'):.2f}")

    # context switch = the CPU changes from one process to a different one
    runs = [name for name, s, e in chart if name != "IDLE"]
    switches = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    makespan = chart[-1][2]

    return {
        "avg_wt": avg("wt"),
        "avg_tat": avg("tat"),
        "avg_rt": avg("rt"),
        "max_wt": max(m["wt"] for m in metrics.values()),
        "switches": switches,
        "makespan": makespan,
        "throughput": n / makespan,
        "chart": chart,                  # list of (pid or "IDLE", start, end) for the Gantt chart
        "per_process": metrics,          # pid -> {at, bt, ct, tat, wt, rt}
    }


def fcfs(processes, verbose=False):
    chart = []
    time = 0
    for p in sorted(processes, key=lambda x: x["at"]):
        start = max(time, p["at"])
        add_segment(chart, "IDLE", time, start)          # BUGFIX: gap is idle, not the process
        add_segment(chart, p["pid"], start, start + p["bt"])
        time = start + p["bt"]
    return finish(processes, chart, verbose)


def sjf(processes, verbose=False):
    """Non-preemptive Shortest Job First (knows the true burst times)."""
    remaining = processes[:]
    chart = []
    time = 0
    while remaining:
        ready = [p for p in remaining if p["at"] <= time]
        if not ready:
            next_at = min(p["at"] for p in remaining)
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        p = min(ready, key=lambda x: (x["bt"], x["at"]))
        add_segment(chart, p["pid"], time, time + p["bt"])
        time += p["bt"]
        remaining.remove(p)
    return finish(processes, chart, verbose)


def srtf(processes, verbose=False):
    """Preemptive SJF: Shortest Remaining Time First."""
    remaining = {p["pid"]: p["bt"] for p in processes}
    chart = []
    time = 0
    completed = 0
    while completed < len(processes):
        ready = [p for p in processes if p["at"] <= time and remaining[p["pid"]] > 0]
        if not ready:
            next_at = min(p["at"] for p in processes if remaining[p["pid"]] > 0)
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        p = min(ready, key=lambda x: (remaining[x["pid"]], x["at"]))
        pid = p["pid"]
        add_segment(chart, pid, time, time + 1)
        remaining[pid] -= 1
        time += 1
        if remaining[pid] == 0:
            completed += 1
    return finish(processes, chart, verbose)


def priority_np(processes, verbose=False):
    """Non-preemptive priority scheduling (lower number = higher priority)."""
    remaining = processes[:]
    chart = []
    time = 0
    while remaining:
        ready = [p for p in remaining if p["at"] <= time]
        if not ready:
            next_at = min(p["at"] for p in remaining)
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        p = min(ready, key=lambda x: (x["priority"], x["at"]))
        add_segment(chart, p["pid"], time, time + p["bt"])
        time += p["bt"]
        remaining.remove(p)
    return finish(processes, chart, verbose)


def priority_p(processes, verbose=False):
    """Preemptive priority scheduling (lower number = higher priority)."""
    remaining = {p["pid"]: p["bt"] for p in processes}
    chart = []
    time = 0
    completed = 0
    while completed < len(processes):
        ready = [p for p in processes if p["at"] <= time and remaining[p["pid"]] > 0]
        if not ready:
            next_at = min(p["at"] for p in processes if remaining[p["pid"]] > 0)
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        p = min(ready, key=lambda x: (x["priority"], x["at"]))
        pid = p["pid"]
        add_segment(chart, pid, time, time + 1)
        remaining[pid] -= 1
        time += 1
        if remaining[pid] == 0:
            completed += 1
    return finish(processes, chart, verbose)


def round_robin(processes, quantum, verbose=False):
    if quantum <= 0:
        raise ValueError("Time quantum must be positive.")
    ordered = sorted(processes, key=lambda p: p["at"])
    remaining = {p["pid"]: p["bt"] for p in processes}
    queue = deque()
    chart = []
    time = 0
    i = 0
    completed = 0
    while completed < len(processes):
        while i < len(ordered) and ordered[i]["at"] <= time:
            queue.append(ordered[i]["pid"])
            i += 1
        if not queue:
            next_at = ordered[i]["at"]
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        pid = queue.popleft()
        run = min(quantum, remaining[pid])
        add_segment(chart, pid, time, time + run)
        time += run
        remaining[pid] -= run
        # arrivals during the slice enter the queue BEFORE the preempted process
        while i < len(ordered) and ordered[i]["at"] <= time:
            queue.append(ordered[i]["pid"])
            i += 1
        if remaining[pid] > 0:
            queue.append(pid)
        else:
            completed += 1
    return finish(processes, chart, verbose)


def mlfq(processes, quantums=(2, 4, 8), verbose=False):
    """
    Multi-level feedback queue. Level 0 is the highest priority.
    quantums[k] is the time slice of level k. A process that uses its whole
    slice moves down one level (the last level is round robin). New arrivals
    enter level 0 and can preempt lower levels. No priority boost.
    """
    levels = len(quantums)
    queues = [deque() for _ in range(levels)]
    ordered = sorted(processes, key=lambda p: p["at"])
    remaining = {p["pid"]: p["bt"] for p in processes}
    chart = []
    time = 0
    i = 0
    completed = 0
    while completed < len(processes):
        while i < len(ordered) and ordered[i]["at"] <= time:
            queues[0].append(ordered[i]["pid"])
            i += 1
        q = next((j for j in range(levels) if queues[j]), None)
        if q is None:
            next_at = ordered[i]["at"]
            add_segment(chart, "IDLE", time, next_at)
            time = next_at
            continue
        pid = queues[q].popleft()
        budget = quantums[q]
        used = 0
        while used < budget and remaining[pid] > 0:
            while i < len(ordered) and ordered[i]["at"] <= time:
                queues[0].append(ordered[i]["pid"])
                i += 1
            if q > 0 and queues[0]:
                break                       # a higher level has work: preempt
            add_segment(chart, pid, time, time + 1)
            time += 1
            used += 1
            remaining[pid] -= 1
        if remaining[pid] == 0:
            completed += 1
        elif used < budget and q > 0 and queues[0]:
            queues[q].append(pid)           # preempted: stays on the same level
        else:
            queues[min(q + 1, levels - 1)].append(pid)   # used full slice: demote
    return finish(processes, chart, verbose)
