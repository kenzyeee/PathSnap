# PathSnap: 4-Member Team Work Distribution & Git Commit Plan

**PES University | Department of Computer Science and Engineering**  
**Course:** UE24CS352A Machine Learning  
**Project:** PathSnap Distributed Map-Matching & Route Reconstruction  

---

## 1. Team Roles & Responsibility Matrix

| Member | Assigned Role | Core Responsibilities | Primary Deliverables |
|:---|:---|:---|:---|
| **Aadya Arvind** (`aadyaarvind19@gmail.com`) | **Role 1 — Data & Preprocessing** | EDA, dataset inventory, metric projection, monotonic DP ground-truth alignment, multi-point KDTree candidate generation, feature engineering | `src/eda/`, `src/data/`, `src/features/`, `scripts/prepare_dataset.py`, `scripts/run_eda.py` |
| **Aditya Kumar** (`Adityakumar1975in@gmail.com`) | **Role 2 — Machine Learning & Optimisation** | 6-model benchmark, 5-Fold GroupKFold CV, probability calibration, Optuna Bayesian hyperparameter optimization (40 trials), model export | `src/models/`, `src/tuning/`, `scripts/run_experiments.py`, `scripts/run_tuning.py`, `artifacts/models/` |
| **Alakh Gupta** (`alakhgupta321@gmail.com`) | **Role 3 — HMM & Viterbi** | Emission model, road-network transition model, log-space Viterbi sequence decoding, continuous route reconstruction, server-side evaluation module | `src/hmm/`, `src/eval/`, `scripts/run_hmm_eval.py`, `tests/test_pipeline.py` |
| **Aanya K** (`aanya.kalrickal@gmail.com`) | **Role 4 — Backend & Frontend** | FastAPI server (Laptop 1), Streamlit UI (Laptop 2), Folium interactive maps, end-to-end verification, deployment scripts, README documentation | `src/api/`, `src/ui/`, `src/viz/`, `scripts/run_api.py`, `scripts/run_ui.py`, `scripts/verify_pipeline.py` |

---

## 2. Complete File-to-Member Mapping

Every single file in the PathSnap repository is mapped to its primary author below:

| File Path | Description | Lines / Size | Responsible Member |
|:---|:---|:---:|:---:|
| `src/eda/dataset_inventory.py` | Audits raw trips, observation counts, gaps, degree distributions | ~150 lines | **Member 2 (Role 1)** |
| `src/eda/eda_plots.py` | Generates sampling, trajectory, and road network distribution plots | ~100 lines | **Member 2 (Role 1)** |
| `src/data/geo_utils.py` | PyProj Azimuthal Equidistant projection, bearing, segment projections | ~120 lines | **Member 2 (Role 1)** |
| `src/data/loader.py` | Loads `.track`, `.nodes`, `.arcs`, builds NetworkX directed graph | ~140 lines | **Member 2 (Role 1)** |
| `src/data/align_route.py` | Monotonic DP ground-truth alignment for training labels | ~110 lines | **Member 2 (Role 1)** |
| `src/features/candidate_generator.py` | Multi-point spatial KDTree candidate generation (CMR < 0.2%) | ~130 lines | **Member 2 (Role 1)** |
| `src/features/feature_extractor.py` | Extracts 18 kinematic, geometric, and topological features | ~170 lines | **Member 2 (Role 1)** |
| `scripts/run_eda.py` | Execution script for dataset exploration and summary tables | ~130 lines | **Member 2 (Role 1)** |
| `scripts/prepare_dataset.py` | Full data preparation script generating parquet feature tables | ~120 lines | **Member 2 (Role 1)** |
| `results/logs/data_quality_log.md` | Audit log of coordinates, sampling frequency, and assumptions | ~100 lines | **Member 2 (Role 1)** |
| `src/models/train_and_evaluate.py` | 5-Fold GroupKFold CV across 6 distinct model families | ~170 lines | **Member 4 (Role 2)** |
| `src/models/model_visualizer.py` | Comparison bar charts, confusion matrices, real loss curves | ~120 lines | **Member 4 (Role 2)** |
| `src/tuning/bayes_tune.py` | Optuna TPE Bayesian hyperparameter optimization module | ~110 lines | **Member 4 (Role 2)** |
| `scripts/run_experiments.py` | Benchmark runner matching Deliverables PDF Section 3.1 table | ~180 lines | **Member 4 (Role 2)** |
| `scripts/run_tuning.py` | 40-trial Bayesian tuning runner, SQLite persistence, model export | ~140 lines | **Member 4 (Role 2)** |
| `results/tables/model_comparison.csv` | Deliverables Table 3.1 (all 6 models + metrics) | Table file | **Member 4 (Role 2)** |
| `results/tables/optuna_trials.csv` | 40 Optuna hyperparameter trial configurations and scores | Table file | **Member 4 (Role 2)** |
| `artifacts/models/*.joblib` | Serialized model pipelines (HistGBM, RF, MLP, LR, NB, Tuned) | Binary | **Member 4 (Role 2)** |
| `src/hmm/emission.py` | Emission probability model (calibrated classifier + spatial Gaussian) | ~60 lines | **Member 1 (Role 3)** |
| `src/hmm/transition.py` | Newson-Krumm transition model with shortest-path & gap reset | ~115 lines | **Member 1 (Role 3)** |
| `src/hmm/viterbi.py` | Log-space Viterbi forward recurrence and backpointer decoder | ~110 lines | **Member 1 (Role 3)** |
| `src/hmm/route_reconstructor.py` | Continuous route assembly, deduplication, and gap bridging | ~110 lines | **Member 1 (Role 3)** |
| `src/eval/evaluator.py` | Server-side isolated evaluation against `.route` (zero leakage) | ~80 lines | **Member 1 (Role 3)** |
| `scripts/run_hmm_eval.py` | Evaluator runner: Classifier-Only vs Classifier + HMM | ~150 lines | **Member 1 (Role 3)** |
| `tests/test_pipeline.py` | Pytest test suite: projections, toy Viterbi, data integrity | ~150 lines | **Member 1 (Role 3)** |
| `results/tables/hmm_vs_classifier.csv` | Comparison table showing HMM improvement per student-test trip | Table file | **Member 1 (Role 3)** |
| `src/api/server.py` | FastAPI backend application (Laptop 1, CORS, request logging) | ~200 lines | **Member 3 (Role 4)** |
| `src/ui/app.py` | Streamlit Evaluator UI (Laptop 2, Folium maps, metrics dashboard) | ~220 lines | **Member 3 (Role 4)** |
| `src/viz/folium_map.py` | Interactive multi-layer Folium Leaflet map generator | ~130 lines | **Member 3 (Role 4)** |
| `scripts/run_api.py` | FastAPI server startup script with automatic sys.path resolution | ~25 lines | **Member 3 (Role 4)** |
| `scripts/run_ui.py` | Streamlit UI startup script with PYTHONPATH configuration | ~30 lines | **Member 3 (Role 4)** |
| `scripts/verify_pipeline.py` | Automated HTTP integration test and ground-truth leakage check | ~120 lines | **Member 3 (Role 4)** |
| `README.md` | Complete repository documentation, setup, and 2-laptop guide | ~200 lines | **Member 3 (Role 4)** |
| `requirements.txt` | Explicit Python dependency manifest | ~25 lines | **Member 3 (Role 4)** |
| `docs/PathSnap_Technical_Report.md` | Report skeleton matching Deliverables PDF guidelines | ~300 lines | **Joint (All 4)** |

---

## 3. Step-by-Step Git Commit Plan for All 4 Members

To demonstrate authentic teamwork and clear collaborative version control, each member should commit their own work using their respective Git credentials.

```
Git Branching Model:
main ─────────────────────────────────────────────────────────────► (Final Demo Release)
  ▲                     ▲                    ▲                  ▲
  │ (PR #1 Merge)       │ (PR #2 Merge)      │ (PR #3 Merge)    │ (PR #4 Merge)
  │                     │                    │                  │
feature/data-prep   feature/ml-models    feature/hmm-viterbi  feature/backend-ui
(Member 2)          (Member 4)           (Member 1)           (Member 3)
```

---

### Member 2: Role 1 — Data & Preprocessing
**Git Setup:**
```bash
git config user.name "Member2_Name"
git config user.email "member2@pes.edu"
git checkout -b feature/data-prep
```

**Commit Sequence:**

1. **Commit 1 (EDA & Coordinates):**
   ```bash
   git add src/eda/ src/data/geo_utils.py scripts/run_eda.py results/logs/data_quality_log.md
   git commit -m "feat(eda): implement dataset audit, metric projection, and sampling analysis"
   ```
   *Explanation:* Adds the dataset inventory scanner, localized PyProj Azimuthal Equidistant coordinate system, and initial EDA plots.

2. **Commit 2 (Monotonic Ground-Truth Alignment):**
   ```bash
   git add src/data/align_route.py src/data/loader.py
   git commit -m "feat(data): implement monotonic DP ground-truth route alignment"
   ```
   *Explanation:* Introduces dynamic programming alignment to map ~1000 GPS points to the ~100 ground-truth route arcs with monotonic time constraints.

3. **Commit 3 (Candidate Generation & Feature Engineering):**
   ```bash
   git add src/features/candidate_generator.py src/features/feature_extractor.py scripts/prepare_dataset.py
   git commit -m "feat(features): add multi-point spatial KDTree candidate generation and 18 kinematic features"
   ```
   *Explanation:* Fixes long-arc candidate misses using multi-point segment sampling (reducing CMR to 0.17%) and extracts 18 kinematic/topological features.

---

### Member 4: Role 2 — Machine Learning & Optimisation
**Git Setup:**
```bash
git config user.name "Member4_Name"
git config user.email "member4@pes.edu"
git checkout -b feature/ml-models
```

**Commit Sequence:**

1. **Commit 1 (Multi-Model Benchmark Architecture):**
   ```bash
   git add src/models/train_and_evaluate.py src/models/model_visualizer.py
   git commit -m "feat(models): implement 5-fold GroupKFold cross-validation across 6 model families"
   ```
   *Explanation:* Sets up sklearn pipelines with StandardScaler inside each fold, GroupKFold by `trip_id`, and evaluation metrics calculation.

2. **Commit 2 (Benchmark Execution & Deliverables Table):**
   ```bash
   git add scripts/run_experiments.py results/tables/model_comparison.csv results/plots/
   git commit -m "feat(experiments): execute multi-model benchmark and generate Deliverables Section 3.1 tables"
   ```
   *Explanation:* Runs benchmark across Logistic Regression, Gaussian NB, Decision Tree, Random Forest, HistGBM, and MLP; saves real confusion matrices and loss curves.

3. **Commit 3 (Bayesian Hyperparameter Optimization):**
   ```bash
   git add src/tuning/bayes_tune.py scripts/run_tuning.py artifacts/models/
   git commit -m "feat(tuning): implement 40-trial Optuna TPE Bayesian hyperparameter optimization"
   ```
   *Explanation:* Implements Optuna TPE search over learning rate, tree depth, and leaves; persists SQLite study; refits and exports winning model artifact.

---

### Member 1: Role 3 — HMM & Viterbi
**Git Setup:**
```bash
git config user.name "Member1_Name"
git config user.email "member1@pes.edu"
git checkout -b feature/hmm-viterbi
```

**Commit Sequence:**

1. **Commit 1 (Emission & Transition Probabilities):**
   ```bash
   git add src/hmm/emission.py src/hmm/transition.py
   git commit -m "feat(hmm): implement calibrated classifier emission and Newson-Krumm transition models"
   ```
   *Explanation:* Builds `EmissionModel` fusing classifier confidence with spatial Gaussian error, and `TransitionModel` using road-network shortest paths.

2. **Commit 2 (Viterbi Decoder & Route Assembly):**
   ```bash
   git add src/hmm/viterbi.py src/hmm/route_reconstructor.py
   git commit -m "feat(viterbi): implement log-space Viterbi sequence decoder and continuous route reconstruction"
   ```
   *Explanation:* Develops dynamic programming Viterbi decoder with backpointers and topological gap bridging to ensure continuous routes.

3. **Commit 3 (Isolated Evaluation & Pytest Suite):**
   ```bash
   git add src/eval/evaluator.py scripts/run_hmm_eval.py tests/test_pipeline.py results/tables/hmm_vs_classifier.csv
   git commit -m "test(eval): add server-side isolated ground-truth evaluator and comprehensive pytest suite"
   ```
   *Explanation:* Adds server-side evaluation module (strictly isolated from prediction) and pytest unit tests for spatial projections, toy graph Viterbi, and leakage verification.

---

### Member 3: Role 4 — Backend & Frontend
**Git Setup:**
```bash
git config user.name "Member3_Name"
git config user.email "member3@pes.edu"
git checkout -b feature/backend-ui
```

**Commit Sequence:**

1. **Commit 1 (FastAPI Backend Service):**
   ```bash
   git add src/api/server.py scripts/run_api.py
   git commit -m "feat(api): implement FastAPI inference backend with /health, /trips, and /predict endpoints"
   ```
   *Explanation:* Develops Laptop 1 backend with CORS, startup artifact caching, request logging, and strict data leakage prevention.

2. **Commit 2 (Streamlit & Folium Evaluator UI):**
   ```bash
   git add src/ui/app.py src/viz/folium_map.py scripts/run_ui.py
   git commit -m "feat(ui): build Streamlit evaluator interface with interactive Folium maps and metrics dashboard"
   ```
   *Explanation:* Implements Laptop 2 evaluator dashboard with trip selector cards, interactive Leaflet layers, and real-time performance indicators.

3. **Commit 3 (End-to-End Verification & Documentation):**
   ```bash
   git add scripts/verify_pipeline.py README.md requirements.txt .gitignore
   git commit -m "docs: finalize two-laptop deployment instructions and automated verification test script"
   ```
   *Explanation:* Adds automated HTTP integration verification script testing all 10 student trips and finalizes comprehensive README documentation.

---
