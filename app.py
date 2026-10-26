"""
Adaptive CPU Scheduler Recommender (Streamlit app).      Run:  streamlit run app.py

1. You enter processes (arrival time, burst time, priority).
2. The trained model looks at the workload and shortlists the most promising schedulers.
3. Only the shortlisted schedulers are simulated; the best one is picked and shown with its Gantt chart.
"""
import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st

from config import FAMILIES, WEIGHTS
from candidates import CLAIRVOYANT
from workloads import make_workload
from recommend import (find_model_path, load_bundle, version_warnings, validate, recommend,
                       metrics_frame, TRAIN_N_RANGE)
from gantt import gantt_figure, gantt_text

st.set_page_config(page_title="Adaptive CPU Scheduler", layout="wide")

EXAMPLES = {
    "Hospital exam question (5 processes, with priorities)":
        [("P1", 0, 7, 3), ("P2", 1, 6, 2), ("P3", 2, 5, 1), ("P4", 5, 3, 2), ("P5", 8, 4, 1)],
    "Textbook example (5 processes, no priorities)":
        [("P1", 0, 8, 0), ("P2", 1, 4, 0), ("P3", 2, 2, 0), ("P4", 3, 6, 0), ("P5", 4, 3, 0)],
}
SRC_RANDOM, SRC_CSV, SRC_BLANK = "Random workload from the generator", "Upload a CSV file", "Blank table"
SOURCES = list(EXAMPLES) + [SRC_RANDOM, SRC_CSV, SRC_BLANK]
COLS = ["pid", "at", "bt", "priority"]
METHODS = {"Random forest classifier (vote share)": "classifier",
           "Random forest regressor (predicted score)": "regressor"}


# ------------------------------------------------------------------ model loading
@st.cache_resource
def _load_path(path, mtime):
    return load_bundle(path)


@st.cache_resource
def _load_bytes(data):
    return load_bundle(io.BytesIO(data))


def get_bundle():
    st.sidebar.header("Model")
    uploaded = st.sidebar.file_uploader("Load a model file (ranker.joblib)", type=["joblib"],
                                        help="Only load files you trained yourself.")
    try:
        if uploaded is not None:
            return _load_bytes(uploaded.getvalue()), f"uploaded file {uploaded.name}"
        path = find_model_path()
        if path:
            import os
            return _load_path(path, os.path.getmtime(path)), path
    except Exception as e:                      # noqa: BLE001 - show any loading problem to the user
        st.sidebar.error(f"Could not load the model: {e}")
        st.stop()
    st.error("No trained model found.")
    st.markdown(
        "Put the model you downloaded from Colab at **`models/ranker.joblib`** and reload this page "
        "(or upload it in the sidebar).\n\n"
        "To just try the interface, run `python make_demo_model.py` in a terminal. It builds a small "
        "demo model in under a minute.")
    st.stop()


# ------------------------------------------------------------------ input helpers
def csv_to_frame(file):
    raw = pd.read_csv(file)
    raw.columns = [c.strip().lower() for c in raw.columns]
    alias = {"pid": ["pid", "process", "name", "id"], "at": ["at", "arrival", "arrival time", "arrival_time"],
             "bt": ["bt", "burst", "burst time", "burst_time"], "priority": ["priority", "prio"]}
    out = pd.DataFrame()
    for key, names in alias.items():
        hit = next((n for n in names if n in raw.columns), None)
        if hit is None and key != "priority":
            raise ValueError(f"CSV needs a column for '{key}' (one of: {', '.join(names)}).")
        out[key] = raw[hit] if hit else 0
    return out


def initial_frame(source, family, seed, csv_file):
    if source in EXAMPLES:
        return pd.DataFrame(EXAMPLES[source], columns=COLS)
    if source == SRC_RANDOM:
        return pd.DataFrame(make_workload(family, int(seed)))[COLS]
    if source == SRC_CSV and csv_file is not None:
        return csv_to_frame(csv_file)
    return pd.DataFrame({"pid": ["P1", "P2", "P3"], "at": [0, 0, 0], "bt": [5, 3, 2], "priority": [0, 0, 0]})


# ------------------------------------------------------------------ page
bundle, model_where = get_bundle()
meta = bundle.get("meta", {})

st.title("Adaptive CPU Scheduler Recommender")
st.caption("Enter a set of processes. A trained model shortlists the most promising scheduling algorithms; "
           "only those are simulated, and the best one is shown with its Gantt chart.")

with st.sidebar:
    st.caption(f"Loaded: {model_where}")
    if meta:
        st.caption(f"Trained on {meta.get('n_dev_workloads', '?')} workloads, "
                   f"scikit-learn {meta.get('sklearn', '?')}, {meta.get('created', '')}")
    st.header("Settings")
    k = st.slider("Shortlist size (top-k)", 1, 5, 3)
    method_label = st.radio("Ranking model", list(METHODS), index=0)
    verify = st.checkbox("Also run every scheduler to check the shortlist", value=True,
                         help="For demonstration: shows whether the shortlist really contained the best scheduler.")
    st.caption("Score = " + " + ".join(f"{w:.2f} x {n}" for n, w in WEIGHTS.items())
               + ". Each term is scaled 0 to 1 across the schedulers being compared (0 = best).")

if meta.get("demo"):
    st.warning("This is a small DEMO model (trained on a few hundred workloads). Use it only to test the "
               "interface. For real results, train on Colab and replace models/ranker.joblib.")
for msg in version_warnings(bundle):
    st.warning(msg)

# ---- 1. input
st.header("1. Enter the processes")
c1, c2, c3 = st.columns([2, 1, 1])
source = c1.selectbox("Start from", SOURCES)
family = seed = csv_file = None
if source == SRC_RANDOM:
    family = c2.selectbox("Family", FAMILIES)
    seed = c3.number_input("Seed", min_value=0, value=0, step=1)
elif source == SRC_CSV:
    csv_file = c2.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    c3.caption("Columns: pid, at, bt, priority (priority optional)")

try:
    start_df = initial_frame(source, family, seed, csv_file)
except Exception as e:                           # noqa: BLE001
    st.error(f"Could not read that input: {e}")
    start_df = initial_frame(SRC_BLANK, None, None, None)

editor_key = f"editor|{source}|{family}|{seed}|{csv_file.name if csv_file else ''}"
edited = st.data_editor(
    start_df, key=editor_key, num_rows="dynamic", width="stretch", hide_index=True,
    column_config={
        "pid": st.column_config.TextColumn("Process"),
        "at": st.column_config.NumberColumn("Arrival time", min_value=0, step=1),
        "bt": st.column_config.NumberColumn("Burst time", min_value=1, step=1),
        "priority": st.column_config.NumberColumn("Priority (lower = higher)", step=1),
    })
st.caption("Times are whole numbers (e.g. milliseconds). Lower priority number = higher priority. "
           "Add or delete rows with the table controls.")

procs, errors, warnings = validate(edited)
for e in errors:
    st.error(e)
for w in warnings:
    st.warning(w)

sig = hashlib.md5(repr((procs, k, method_label, verify)).encode()).hexdigest()
go = st.button("Recommend schedulers", type="primary", disabled=bool(errors))
if go:
    with st.spinner("Ranking schedulers and simulating the shortlist..."):
        st.session_state["result"] = recommend(bundle, procs, k=k, method=METHODS[method_label], verify=verify)
        st.session_state["sig"] = sig
        st.session_state["procs"] = procs

res = st.session_state.get("result")
if res is None:
    st.stop()
if st.session_state.get("sig") != sig:
    st.info("The input or settings changed since the last run. Press **Recommend schedulers** to refresh the results below.")

procs = st.session_state["procs"]
feats = res["features"]
winner = res["winner"]

# ---- 2. workload summary
st.header("2. Workload summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Processes", feats["n"])
m2.metric("Average burst time", f"{feats['burst_mean']:.2f}")
m3.metric("Load (work per unit arrival time)", f"{feats['load']:.2f}",
          help="Above 1 means the work arrives faster than one CPU can finish it, so a queue builds up.")
m4.metric("Priority vs burst correlation", f"{feats['prio_burst_corr']:.2f}",
          help="Positive: low-priority-number (important) jobs tend to be short.")
with st.expander("All 10 features given to the model"):
    st.dataframe(pd.DataFrame({"feature": list(feats), "value": [round(float(v), 4) for v in feats.values()]}),
                 hide_index=True, width="stretch")

# ---- 3. model shortlist
st.header(f"3. The model's shortlist (top {res['k']})")
rk = res["ranking"].copy()
rk["in shortlist"] = np.where(rk["rank"] <= res["k"], "yes", "")
st.dataframe(
    rk.rename(columns={"scheduler": "Scheduler", "vote_share": "Vote share", "predicted_score": "Predicted score",
                       "rank": "Rank"}),
    hide_index=True, width="stretch",
    column_config={"Vote share": st.column_config.ProgressColumn("Vote share", min_value=0.0, max_value=1.0,
                                                                 format="%.2f"),
                   "Predicted score": st.column_config.NumberColumn("Predicted score", format="%.3f")})
st.caption("Vote share = fraction of the forest's trees that picked this scheduler as best. Predicted score = the "
           "regressor's guess of its score (lower is better). Ranking uses: "
           + ("vote share." if res["method"] == "classifier" else "predicted score."))

# ---- 4. simulate the shortlist
st.header("4. Simulating only the shortlisted schedulers")
tab = metrics_frame(res["results"], res["scores"]).sort_values("score").reset_index(drop=True)
tab.insert(1, "result", np.where(tab["scheduler"] == winner, "BEST", ""))
show = tab.rename(columns={"scheduler": "Scheduler", "result": "", "avg_wt": "Avg waiting", "avg_tat": "Avg turnaround",
                           "avg_rt": "Avg response", "max_wt": "Max waiting", "switches": "Context switches",
                           "score": "Score"})
st.dataframe(show.style.format({"Avg waiting": "{:.2f}", "Avg turnaround": "{:.2f}", "Avg response": "{:.2f}",
                                "Score": "{:.3f}"}).apply(
    lambda r: ["background-color: #d4edda; color: #111" if r[""] == "BEST" else "" for _ in r], axis=1),
    hide_index=True, width="stretch")
st.caption("Each metric is first scaled 0 (best of the schedulers in this table) to 1 (worst), then weighted. "
           "So a score of 0 would mean best on every metric and 1 worst on every metric; lower is better. "
           "Scores are only comparable within one table, not across runs.")
st.download_button("Download this table (CSV)", tab.to_csv(index=False), file_name="shortlist_results.csv")

# ---- 5. best scheduler
st.header(f"5. Best scheduler: {winner}")
view = st.selectbox("Show details for", res["shortlist"], index=res["shortlist"].index(winner),
                    format_func=lambda n: f"{n} (best)" if n == winner else n)
r = res["results"][view]
d1, d2, d3, d4 = st.columns(4)
d1.metric("Average waiting time", f"{r['avg_wt']:.2f}")
d2.metric("Average turnaround time", f"{r['avg_tat']:.2f}")
d3.metric("Average response time", f"{r['avg_rt']:.2f}")
d4.metric("Context switches", r["switches"])

style = st.radio("Gantt chart style", ["auto", "row", "rows"], horizontal=True,
                 format_func={"auto": "Auto", "row": "Textbook (one row)", "rows": "One row per process"}.get,
                 help="Auto uses the textbook style for small workloads and one row per process for large ones.")
st.pyplot(gantt_figure(r["chart"], title=f"Gantt chart: {view}", style=style), width="stretch")
with st.expander("Text version of the Gantt chart"):
    st.code(gantt_text(r["chart"]), language=None)

pp = pd.DataFrame(r["per_process"]).T
prio = {p["pid"]: p["priority"] for p in procs}
pp.insert(3, "priority", [prio[i] for i in pp.index])
pp = pp.rename(columns={"at": "Arrival", "bt": "Burst", "ct": "Completion", "tat": "Turnaround",
                        "wt": "Waiting", "rt": "Response", "priority": "Priority"})
pp.index.name = "Process"
order = {p["pid"]: i for i, p in enumerate(procs)}
pp = pp.loc[sorted(pp.index, key=order.get)]
st.subheader("Per-process results")
st.dataframe(pp, width="stretch")
st.caption(f"Averages: waiting {r['avg_wt']:.2f}, turnaround {r['avg_tat']:.2f}, response {r['avg_rt']:.2f}. "
           f"Waiting = turnaround - burst. Response = first run - arrival.")

# ---- 6. check against everything
if res["verify"]:
    v = res["verify"]
    st.header("6. Check: what if every scheduler had been simulated?")
    if v["shortlist_has_best"]:
        st.success(f"The shortlist contained a best scheduler ({', '.join(v['best_names'])}). "
                   f"Score gap of the chosen scheduler to the true best: {v['winner_regret']:.3f}.")
    else:
        st.warning(f"The shortlist did NOT contain the best scheduler ({', '.join(v['best_names'])}). "
                   f"The chosen scheduler's score gap to the true best is {v['winner_regret']:.3f} "
                   f"(0 would be perfect).")
    rows = []
    for n, rr in v["all_results"].items():
        rows.append({"Scheduler": n, "Avg waiting": rr["avg_wt"], "Avg response": rr["avg_rt"],
                     "Max waiting": rr["max_wt"], "Context switches": rr["switches"],
                     "Score (all 10)": v["pool_scores"].get(n, np.nan),
                     "Note": ("needs true burst times: reference only" if n in CLAIRVOYANT else
                              ("in shortlist" if n in res["shortlist"] else ""))})
    allt = pd.DataFrame(rows).sort_values("Score (all 10)", na_position="last").reset_index(drop=True)
    st.dataframe(allt.style.format({"Avg waiting": "{:.2f}", "Avg response": "{:.2f}", "Score (all 10)": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("Here scores are scaled across all 10 candidate schedulers. SJF and SRTF are shown for reference only: "
               "they need the true burst times in advance, which a real operating system does not have, so they "
               "are not allowed to win.")

st.divider()
st.caption("Limits: the model was trained on synthetic workloads of "
           f"{TRAIN_N_RANGE[0]}-{TRAIN_N_RANGE[1]} processes and the 'best' scheduler is defined by the score above, "
           "which is a design choice. The simulation uses one CPU, one burst per process and no I/O. "
           "Treat the shortlist as a suggestion, and the simulated numbers as exact for the processes you entered.")
