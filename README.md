# Adaptive CPU Scheduler: ML shortlist project (v2)

**Question:** can a machine-learning model look at a workload and shortlist the 2-3 scheduling
algorithms most likely to be best, so only those need to be simulated?

```
processes -> 10 feature numbers -> ML ranks the schedulers -> keep top-k
          -> simulate ONLY those k -> pick the best by score -> Gantt chart + metrics
```

## Where things run

| Where | What |
|---|---|
| **Google Colab** | generate the big dataset (`colab/01_generate_data.ipynb`), merge + train the model (`colab/02_merge_and_train.ipynb`) |
| **Your Mac** | the code (Git), tests, and the app (`app.py`), which loads the downloaded `models/ranker.joblib` |

Heavy work stays off the laptop. Nothing on the Mac needs more than a few seconds of CPU.

## Quick start on the Mac (to see the interface right away)

Needs Python 3.10 or newer (check with `python3 --version`; install from python.org if older).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-app.txt
python test_schedulers.py          # 7 tests, must all pass
python make_demo_model.py          # tiny DEMO model, ~10 seconds, for trying the interface only
streamlit run app.py               # opens in your browser
```
The demo model is trained on a few hundred workloads: it is for testing the interface, not for results.
The app shows a warning while a demo model is loaded.

## Team workflow

1. **Freeze the code.** Make all code changes first. If you change `schedulers.py`, `workloads.py`,
   `features.py` or `candidates.py`, set `SIM_VERSION = 2` in `config.py`, run both test files, push to
   GitHub, then tag it so everyone uses identical code:
   ```bash
   git tag data-v1 && git push origin data-v1
   ```
2. **Generate data (every teammate, on Colab).** Upload `colab/01_generate_data.ipynb` to Colab (File > Upload
   notebook), edit the settings cell, Run all. Each person uses their own seed block:

   | Person | SEED_START | PER_FAMILY |
   |---|---|---|
   | p1 | 0 | 2000 |
   | p2 | 10000 | 2000 |
   | p3 | 20000 | 2000 |
   | one person: final test set | 900000 | 100 |

   Each CSV lands in your Google Drive (`sched_data/part_<name>.csv`). Everyone shares their file with
   the person who trains. Seeds from 900000 up are the reserved final test set: nobody trains on them.
3. **Train (one person, on Colab).** Put all part files in one Drive folder, open
   `colab/02_merge_and_train.ipynb`, Run all. It merges, checks the labels, trains, prints the results
   tables and downloads `ranker.joblib`. Set `RUN_FINAL_TEST = True` once, at the very end.
4. **Use the model locally.** Put `ranker.joblib` into `models/` and run `streamlit run app.py`.
   The app warns you if scikit-learn versions differ between Colab and your Mac. If so, install the
   version it names (`pip install scikit-learn==X.Y.Z`).

## Using the app

1. Choose a starting point (example, random generator workload, CSV upload, or blank) and edit the table.
   Times are whole numbers; lower priority number = higher priority.
2. Press **Recommend schedulers**. The app shows the workload summary, the model's ranking, the simulated
   shortlist with a score, the best scheduler with its **Gantt chart** (textbook style or one row per
   process), per-process waiting/turnaround/response times, and an optional check against running every
   scheduler.
3. CSV upload format: columns `pid, at, bt, priority` (priority optional).

## Files

| File | Purpose |
|---|---|
| `app.py`, `recommend.py`, `gantt.py` | the user interface, its logic, the Gantt chart drawing |
| `schedulers.py`, `candidates.py` | the algorithms and the list the model chooses from |
| `workloads.py`, `features.py`, `scoring.py`, `config.py` | workload generator, 10 features, score and label, shared settings |
| `generate_data.py`, `merge_data.py`, `inspect_data.py`, `train.py`, `modeling.py` | data and training pipeline |
| `make_demo_model.py` | small demo model for trying the app |
| `demo_cli.py` | the original interactive console menu |
| `test_schedulers.py`, `test_recommend.py` | tests |
| `colab/*.ipynb` | notebooks for data generation and training on Colab |

## Design decisions (state these in the report)

* **Best = lowest weighted score**: 0.45 waiting + 0.25 response + 0.15 worst-case waiting + 0.15 context
  switches, each scaled 0 to 1 across the schedulers being compared. The weights are our choice. Turnaround
  and throughput are left out on purpose: turnaround = waiting + burst, and all schedulers here keep the CPU
  busy whenever a job is ready, so they add no information.
* **SJF and SRTF cannot win** (`CLAIRVOYANT` in `config.py`). They need true burst times, which a real OS
  never has, and SRTF would win most workloads. They are still simulated and shown as reference.
* **Shortlist-then-verify:** the app simulates only the top-k schedulers and picks the best of those.
  The "check" section runs everything to show whether the shortlist contained the true best.
* **Offline tool:** features use the whole workload, so this is not a live kernel scheduler.

## Limits to be honest about

* Trained on synthetic workloads of 10 to 60 processes from 6 hand-designed families. Smaller or very
  different workloads are outside what the model has seen. The app warns, and the simulation results are
  still exact.
* "Best" depends on the score weights. Near-ties are common, especially for small workloads.
* The simulator is single-CPU, one burst per process, no I/O, whole-number times, and context switches cost
  no time (they are only counted).
* Only quote results you measured on the merged dataset, in the report, CV or LinkedIn.

## Git notes

`data/*.csv` and `models/*.joblib` are in `.gitignore` (large). Share data through Drive. A model trained
on a few thousand workloads was about 21 MB here, and larger datasets give larger files. GitHub warns above
50 MB and blocks above 100 MB, so check the size before committing a model.
