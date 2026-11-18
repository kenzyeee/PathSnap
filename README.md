# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University | Department of Computer Science and Engineering**  
**Course:** UE24CS352A Machine Learning  
**Project:** End-to-End Distributed Map-Matching and Sequential Trajectory Reconstruction System  

---

## 1. Challenge Overview

GPS tracking systems inherently produce noisy, irregular position observations caused by urban canyon multipath, atmospheric interference, and low sampling frequencies. A vehicle traversing a road network may appear to drift across building footprints or jump between parallel roadways. 

**PathSnap** is an end-to-end, distributed machine learning and hidden Markov model (HMM) system that accurately matches noisy GPS observations to physical road segments and reconstructs topologically continuous, navigable routes.

### Key Performance Highlights
- **Candidate Miss Rate (CMR):** **0.17%** on student testing data (using multi-point spatial KDTree).
- **Map-Matching Top-1 Accuracy:** **96.8%–97.5%** after HMM Viterbi decoding.
- **Route Continuity:** **98.5%+** with topological graph shortest-path gap bridging.
- **Sub-Second Latency:** **< 150 ms** end-to-end inference per trip (serving ~1,500 GPS observations).

---

## 2. Two-Laptop Deployment Architecture

The system is engineered for distributed demonstration across two machines on the local network (LAN):

```
+-----------------------------------------------------------+
|                         LAPTOP 1                          |
|             (Backend Server & Model Host)                 |
|                                                           |
|   +-------------------+        +----------------------+   |
|   |  FastAPI Backend  | <----> | Trained ML Classifier|   |
|   |  (port 8000)      |        | & Calibrated Pipeline|   |
|   +-------------------+        +----------------------+   |
|             ^                             ^               |
|             |                             |               |
|             v                             v               |
|   +-------------------+        +----------------------+   |
|   | Local Road Network|        | Log-Space Viterbi HMM|   |
|   | Metric Projections|        | Sequence Decoder     |   |
|   +-------------------+        +----------------------+   |
+-----------------------------------------------------------+
                              ^
                              | HTTP JSON Requests (LAN)
                              v
+-----------------------------------------------------------+
|                         LAPTOP 2                          |
|                 (Evaluator Interface)                     |
|                                                           |
|   +-------------------+        +----------------------+   |
|   | Streamlit UI      | <----> | Interactive Folium   |   |
|   | (port 8501)       |        | Multi-Layer Map      |   |
|   +-------------------+        +----------------------+   |
|             |                                             |
|             +---> Trip Selector (10 Reserved Trips)       |
|             +---> Performance & Continuity Dashboard      |
+-----------------------------------------------------------+
```

### Separation of Concerns & Integrity
- **Laptop 1 (Model and API):** Hosts trained classifier pipelines, spatial indices, the log-space Viterbi HMM decoder, and the isolated server-side evaluation module.
- **Laptop 2 (User Interface):** Lightweight evaluator UI built with Streamlit and Folium. Evaluators select any of the 10 reserved trips and view real-time predictions and maps.
- **Strict No-Leakage Guarantee:** The prediction pipeline (`/predict`) never accesses or reads `.route`. Ground-truth evaluation remains strictly server-side.

---

## 3. Repository Structure

```
├── README.md                      <- Project documentation and demo day guide
├── requirements.txt               <- Complete Python dependency list
├── .gitignore                     <- Excludes raw data and local environment
├── artifacts/
│   └── models/                    <- Serialized ML models (.joblib)
├── docs/                          <- Hackathon guideline documents
│   ├── Main Guidelines.pdf
│   ├── Dataset Exploration.pdf
│   ├── Models and Training Guidelines.pdf
│   ├── Deliverables.pdf
│   ├── Assessment and Plagiarism Policy.pdf
│   └── PathSnap_Technical_Report.md
├── results/
│   ├── tables/                    <- Consolidated CSV results & comparisons
│   ├── plots/                     <- Real evaluation charts and loss curves
│   ├── maps/                      <- Interactive Folium HTML visualizations
│   └── logs/                      <- Detailed logs and Optuna study SQLite DB
├── src/
│   ├── eda/                       <- Inventory and exploration modules
│   ├── data/                      <- Loaders, coordinate projections, alignment
│   ├── features/                  <- Spatial candidate generation & features
│   ├── models/                    <- Multi-model training and evaluation
│   ├── tuning/                    <- Optuna Bayesian hyperparameter search
│   ├── hmm/                       <- Log-space Viterbi and route reconstruction
│   ├── eval/                      <- Isolated ground-truth evaluation module
│   ├── viz/                       <- Folium map generation
│   ├── api/                       <- FastAPI backend service
│   └── ui/                        <- Streamlit evaluator dashboard
├── scripts/
│   ├── run_eda.py                 <- Runs complete Phase 2 EDA
│   ├── prepare_dataset.py         <- Generates candidate datasets
│   ├── run_experiments.py         <- 5-Fold GroupKFold benchmark on 6 models
│   ├── run_tuning.py              <- Optuna Bayesian optimization (40 trials)
│   ├── run_hmm_eval.py            <- Evaluates Classifier vs HMM decoding
│   ├── run_api.py                 <- Launches FastAPI server
│   ├── run_ui.py                  <- Launches Streamlit UI
│   └── verify_pipeline.py         <- End-to-end integration and leakage tests
└── tests/
    └── test_pipeline.py           <- Pytest unit and integration test suite
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.11 (recommended)
- OpenMP runtime (for Apple Silicon: `brew install libomp`)

### 1. Clone & Set Up Virtual Environment
```bash
git clone <your-repo-url>
cd pathsnap

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. End-to-End Execution Guide

All experiments are 100% reproducible with deterministic random seeds (`seed=42`).

### Step 1: Run Exploratory Data Analysis (EDA)
```bash
PYTHONPATH=. python scripts/run_eda.py
```
*Generates dataset inventory CSVs, 4 distribution plots, 3 interactive Folium maps, and `results/logs/data_quality_log.md`.*

### Step 2: Prepare Candidate Sets and Features
```bash
PYTHONPATH=. python scripts/prepare_dataset.py
```
*Processes all 80 train trips and 10 student-test trips. Outputs compressed feature parquet datasets to `data/processed/`.*

### Step 3: Run Multi-Model Benchmark (6 Model Families)
```bash
PYTHONPATH=. python scripts/run_experiments.py
```
*Executes 5-Fold GroupKFold across Logistic Regression, Gaussian NB, Decision Tree, Random Forest, HistGradientBoosting, and MLP. Outputs `results/tables/model_comparison.csv` matching Deliverables PDF Section 3.1.*

### Step 4: Bayesian Hyperparameter Optimization (Optuna)
```bash
PYTHONPATH=. python scripts/run_tuning.py
```
*Runs 40 trials of TPE search on the 80 train trips. Refits best model and saves `artifacts/models/best_tuned_model.joblib`.*

### Step 5: HMM Viterbi Route Reconstruction Evaluation
```bash
PYTHONPATH=. python scripts/run_hmm_eval.py
```
*Evaluates Classifier-Only vs Classifier + HMM decoding across all 10 student-test trips.*

### Step 6: Run Test Suite & Leakage Verification
```bash
PYTHONPATH=. pytest tests/test_pipeline.py
```

---

## 6. Two-Laptop Demo Day Deployment

### On Laptop 1 (Backend Server)
1. Determine Laptop 1's local IP address:
   - macOS / Linux: `ifconfig | grep "inet " | grep -v 127.0.0.1`
   - Windows: `ipconfig`
   *(Example IP: `192.168.1.105`)*

2. Start the FastAPI backend:
   ```bash
   source .venv/bin/activate
   API_HOST=0.0.0.0 API_PORT=8000 python scripts/run_api.py
   ```
3. Keep this terminal visible to display live inference logs to evaluators.

### On Laptop 2 (Evaluator UI)
1. Ensure Laptop 2 is on the same Wi-Fi / LAN network.
2. Launch the Streamlit dashboard:
   ```bash
   source .venv/bin/activate
   API_BASE_URL="http://192.168.1.105:8000" python scripts/run_ui.py
   ```
3. Open `http://localhost:8501` in your browser.
4. Select any evaluation trip and click **🚀 Run PathSnap**.

---

## 7. Deliverables Checklist Compliance

| Deliverables PDF Requirement | Implementation / File Location | Status |
|:---|:---|:---:|
| Dataset Inventory & File Audit | `src/eda/dataset_inventory.py`, `results/tables/dataset_inventory.csv` | Passed |
| Non-degree Metric Projection | `src/data/geo_utils.py` (Local PyProj AEQD projection) | Passed |
| Minimum 5 Model Families Evaluated | 6 Models evaluated: Linear, NB, Tree, Bagging, Boosting, MLP | Passed |
| GroupKFold by Trip ID | `src/models/train_and_evaluate.py` (5-fold, zero leakage) | Passed |
| Consolidated Model Comparison Table | `results/tables/model_comparison.csv` (PDF Section 3.1 columns) | Passed |
| Real Training Loss Curves | `results/plots/loss_curve_*.png` (Epoch histories for iterative models only) | Passed |
| Bayesian Optimization (Optuna TPE) | `src/tuning/bayes_tune.py` (40 trials, SQLite study persisted) | Passed |
| Log-Space Viterbi HMM Decoder | `src/hmm/viterbi.py` (Newson-Krumm transition + emission model) | Passed |
| Continuous Route Reconstruction | `src/hmm/route_reconstructor.py` (Shortest-path gap bridging) | Passed |
| FastAPI Backend on Laptop 1 | `src/api/server.py` (`0.0.0.0`, request logging, CORS) | Passed |
| Streamlit Evaluator UI on Laptop 2 | `src/ui/app.py` (Folium map, layer controls, metrics) | Passed |
| Strict Server-Side Evaluation Separation | `src/eval/evaluator.py` (Prediction pipeline never reads `.route`) | Passed |
| End-to-End Verification Script | `scripts/verify_pipeline.py` (Automated HTTP assertions) | Passed |
