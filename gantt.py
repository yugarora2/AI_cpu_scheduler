"""Gantt chart drawing (matplotlib)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def gantt_figure(chart, title="", style="auto"):
    """chart: list of (pid or 'IDLE', start, end).
    style 'row'  = textbook single-row chart with labels
          'rows' = one row per process (better when there are many processes)
          'auto' = row if there are few segments, otherwise rows"""
    pids = sorted({p for p, _, _ in chart if p != "IDLE"}, key=_natural)
    cmap = plt.get_cmap("tab20")
    color = {pid: cmap(i % 20) for i, pid in enumerate(pids)}
    color["IDLE"] = "#d9d9d9"
    total = chart[-1][2] - chart[0][1]
    if style == "auto":
        style = "row" if len(chart) <= 40 else "rows"

    if style == "row":
        fig, ax = plt.subplots(figsize=(12, 2.4))
        for pid, s, e in chart:
            ax.barh(0, e - s, left=s, height=0.6, color=color[pid], edgecolor="black", linewidth=0.8,
                    hatch="//" if pid == "IDLE" else None)
            if (e - s) >= total * 0.025:
                ax.text((s + e) / 2, 0, "idle" if pid == "IDLE" else pid, ha="center", va="center", fontsize=9)
        ticks = sorted({s for _, s, _ in chart} | {e for _, _, e in chart})
        if len(ticks) <= 30:
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
    else:
        fig, ax = plt.subplots(figsize=(12, min(16, max(3, 0.3 * len(pids) + 1.2))))
        row = {pid: i for i, pid in enumerate(pids)}
        for pid, s, e in chart:
            if pid == "IDLE":
                continue
            ax.barh(row[pid], e - s, left=s, height=0.7, color=color[pid], edgecolor="none")
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels(pids, fontsize=7)
        ax.invert_yaxis()

    ax.set_xlim(chart[0][1], chart[-1][2])
    ax.set_xlabel("time")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


from schedulers import text_gantt as gantt_text   # same text Gantt chart as the console version
