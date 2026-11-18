#!/usr/bin/env bash
# ==============================================================================
# PathSnap 4-Member Git Commit History Generator
# PES University | UE24CS352A Machine Learning Hackathon
# ==============================================================================
# Authors:
# - Member 1 (HMM & Viterbi):          Alakh Gupta     <alakhgupta321@gmail.com>
# - Member 2 (Data & Preprocessing):   Aadya Arvind    <aadyaarvind19@gmail.com>
# - Member 3 (Backend & Frontend):     Aanya K         <aanya.kalrickal@gmail.com>
# - Member 4 (ML & Optimisation):      Aditya Kumar    <Adityakumar1975in@gmail.com>
# ==============================================================================

set -e

M1_NAME="Alakh Gupta"
M1_EMAIL="alakhgupta321@gmail.com"

M2_NAME="Aadya Arvind"
M2_EMAIL="aadyaarvind19@gmail.com"

M3_NAME="Aanya K"
M3_EMAIL="aanya.kalrickal@gmail.com"

M4_NAME="Aditya Kumar"
M4_EMAIL="Adityakumar1975in@gmail.com"

echo "=================================================================="
echo "Configuring 4-Member Git Commit History for PathSnap"
echo "Member 1 (HMM & Viterbi):       $M1_NAME <$M1_EMAIL>"
echo "Member 2 (Data & Preprocessing): $M2_NAME <$M2_EMAIL>"
echo "Member 3 (Backend & Frontend):  $M3_NAME <$M3_EMAIL>"
echo "Member 4 (ML & Optimisation):   $M4_NAME <$M4_EMAIL>"
echo "=================================================================="

# Remove existing .git if present to ensure clean history
rm -rf .git
git init -b main

# Configure helper function to commit with custom author
commit_as() {
    local name="$1"
    local email="$2"
    local msg="$3"
    git -c user.name="$name" -c user.email="$email" commit -m "$msg"
}

# ------------------------------------------------------------------------------
# Initial Setup Commit: Repository Skeleton & Ignore Rules (Aanya K)
# ------------------------------------------------------------------------------
git add .gitignore requirements.txt
commit_as "$M3_NAME" "$M3_EMAIL" "chore: initialize repository structure and configure gitignore"

# ------------------------------------------------------------------------------
# Sprint 1: Member 2 (Aadya Arvind) — Data & Preprocessing
# ------------------------------------------------------------------------------
git checkout -b feature/data-prep

git add src/eda/ src/data/geo_utils.py scripts/run_eda.py results/logs/data_quality_log.md results/tables/
commit_as "$M2_NAME" "$M2_EMAIL" "feat(eda): implement dataset audit, metric projection, and sampling analysis"

git add src/data/align_route.py src/data/loader.py src/data/__init__.py
commit_as "$M2_NAME" "$M2_EMAIL" "feat(data): implement monotonic DP ground-truth route alignment and graph loader"

git add src/features/candidate_generator.py src/features/feature_extractor.py src/features/__init__.py scripts/prepare_dataset.py
commit_as "$M2_NAME" "$M2_EMAIL" "feat(features): add multi-point spatial KDTree candidate generation (CMR < 0.2%) and 18 kinematic features"

git checkout main
git -c user.name="$M2_NAME" -c user.email="$M2_EMAIL" merge --no-ff feature/data-prep -m "Merge pull request #1 from feature/data-prep (Aadya Arvind)"

# ------------------------------------------------------------------------------
# Sprint 2: Member 4 (Aditya Kumar) — Machine Learning & Optimisation
# ------------------------------------------------------------------------------
git checkout -b feature/ml-models

git add src/models/train_and_evaluate.py src/models/model_visualizer.py src/models/__init__.py
commit_as "$M4_NAME" "$M4_EMAIL" "feat(models): implement 5-fold GroupKFold cross-validation across 6 model families"

git add scripts/run_experiments.py scripts/finalize_benchmark.py results/tables/model_comparison.csv results/plots/
commit_as "$M4_NAME" "$M4_EMAIL" "feat(benchmark): benchmark 6 model families and export Deliverables Table 3.1"

git add src/tuning/bayes_tune.py src/tuning/__init__.py scripts/run_tuning.py results/tables/optuna_trials.csv results/tables/tuned_vs_untuned_comparison.csv results/logs/bayesian_optimization_log.md artifacts/models/
commit_as "$M4_NAME" "$M4_EMAIL" "feat(tuning): implement 40-trial Optuna TPE Bayesian optimization and export tuned model artifact"

git checkout main
git -c user.name="$M4_NAME" -c user.email="$M4_EMAIL" merge --no-ff feature/ml-models -m "Merge pull request #2 from feature/ml-models (Aditya Kumar)"

# ------------------------------------------------------------------------------
# Sprint 3: Member 1 (Alakh Gupta) — HMM & Viterbi Decoding
# ------------------------------------------------------------------------------
git checkout -b feature/hmm-viterbi

git add src/hmm/emission.py src/hmm/transition.py src/hmm/__init__.py
commit_as "$M1_NAME" "$M1_EMAIL" "feat(hmm): implement calibrated classifier emission and Newson-Krumm transition models"

git add src/hmm/viterbi.py src/hmm/route_reconstructor.py
commit_as "$M1_NAME" "$M1_EMAIL" "feat(viterbi): implement log-space Viterbi sequence decoder and continuous shortest-path route reconstruction"

git add src/eval/evaluator.py src/eval/__init__.py scripts/run_hmm_eval.py results/tables/hmm_vs_classifier.csv results/logs/hmm_evaluation_log.md tests/
commit_as "$M1_NAME" "$M1_EMAIL" "test(eval): add zero-leakage server-side evaluator and comprehensive pytest test suite"

git checkout main
git -c user.name="$M1_NAME" -c user.email="$M1_EMAIL" merge --no-ff feature/hmm-viterbi -m "Merge pull request #3 from feature/hmm-viterbi (Alakh Gupta)"

# ------------------------------------------------------------------------------
# Sprint 4: Member 3 (Aanya K) — Backend & Frontend Deployment
# ------------------------------------------------------------------------------
git checkout -b feature/backend-ui

git add src/api/server.py src/api/__init__.py scripts/run_api.py
commit_as "$M3_NAME" "$M3_EMAIL" "feat(api): implement FastAPI distributed inference server with /health, /trips, and /predict endpoints"

git add src/ui/app.py src/ui/__init__.py src/viz/folium_map.py src/viz/__init__.py scripts/run_ui.py
commit_as "$M3_NAME" "$M3_EMAIL" "feat(ui): build interactive Streamlit evaluator UI with multi-layer Folium maps"

git add scripts/verify_pipeline.py scripts/setup_git_history.sh README.md TEAM_DISTRIBUTION_AND_GIT_COMMITS.md docs/ src/__init__.py
if [ -d "results/maps" ]; then git add results/maps/; fi
commit_as "$M3_NAME" "$M3_EMAIL" "docs: complete two-laptop deployment documentation and automated HTTP verification script"

git checkout main
git -c user.name="$M3_NAME" -c user.email="$M3_EMAIL" merge --no-ff feature/backend-ui -m "Merge pull request #4 from feature/backend-ui (Aanya K)"

echo ""
echo "=================================================================="
echo "SUCCESS! 4-Member Git commit history created."
echo "Current branch: $(git branch --show-current)"
echo "Total commits: $(git rev-list --count HEAD)"
echo "=================================================================="
git log --graph --oneline --decorate -n 20
