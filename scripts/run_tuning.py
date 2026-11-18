"""
Bayesian Hyperparameter Optimization Runner for PathSnap.

Executes Optuna study on the best model family (HistGradientBoosting / LightGBM)
using 5-Fold GroupKFold on the 80 train trips.
Saves trials table, optimization history plot, parameter importance plot,
refits best model on all training data, and evaluates on student-test trips.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import optuna
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

from src.features.feature_extractor import FEATURE_COLUMNS
from src.tuning.bayes_tune import run_bayesian_optimization, plot_optuna_diagnostics
from src.models.train_and_evaluate import evaluate_top1_and_continuity
from scripts.run_experiments import load_all_trip_arcs


def main():
    print("=" * 60)
    print("STARTING PHASE 6: BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    data_dir = "data/Student folder"
    processed_dir = "data/processed"
    tables_dir = "results/tables"
    plots_dir = "results/plots"
    logs_dir = "results/logs"
    artifacts_dir = "artifacts/models"

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_dataset.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "student_test_dataset.parquet"))

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    groups_train = train_df["trip_id"].values
    meta_train = train_df[["trip_id", "obs_idx", "candidate_arc_id"]].copy()

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values
    meta_test = test_df[["trip_id", "obs_idx", "candidate_arc_id"]].copy()

    trip_arcs = load_all_trip_arcs(data_dir)

    # 2. Run Optuna Study (40 trials)
    storage_path = os.path.join(logs_dir, "optuna_study.db")
    storage_url = f"sqlite:///{storage_path}"
    
    t0 = time.time()
    study = run_bayesian_optimization(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        meta_train=meta_train,
        trip_arcs=trip_arcs,
        n_trials=40,
        storage_url=storage_url
    )
    study_duration_s = time.time() - t0
    print(f"\nOptuna study completed in {study_duration_s:.1f}s ({study_duration_s/60.0:.2f} min)")

    # 3. Save Trials CSV & Best Parameters
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(tables_dir, "optuna_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Trials table saved: {trials_csv}")

    best_params = study.best_params
    best_value = study.best_value
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best CV Top-1 Accuracy: {best_value * 100.0:.2f}%")
    print(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}")

    # 4. Generate Diagnostic Plots
    plot_optuna_diagnostics(study, plots_dir)
    print("Optimization plots generated in results/plots/")

    # 5. Refit Untuned vs Tuned Model on Full Training Data
    print("\nRefitting untuned baseline model...")
    untuned_clf = HistGradientBoostingClassifier(random_state=42)
    t_tr_un = time.time()
    untuned_clf.fit(X_train, y_train)
    untuned_fit_time = time.time() - t_tr_un

    t_inf_un = time.time()
    untuned_probs = untuned_clf.predict_proba(X_test)[:, 1]
    untuned_infer_ms = (time.time() - t_inf_un) / 10.0 * 1000.0

    untuned_preds = (untuned_probs >= 0.5).astype(int)
    untuned_macro_f1 = f1_score(y_test, untuned_preds, average="macro", zero_division=0)
    untuned_top1, untuned_cont = evaluate_top1_and_continuity(untuned_probs, meta_test, trip_arcs, y_test)
    untuned_ll = log_loss(y_test, untuned_probs)

    print("Refitting tuned model with best parameters...")
    tuned_clf = HistGradientBoostingClassifier(**best_params, random_state=42)
    t_tr_tu = time.time()
    tuned_clf.fit(X_train, y_train)
    tuned_fit_time = time.time() - t_tr_tu

    t_inf_tu = time.time()
    tuned_probs = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_infer_ms = (time.time() - t_inf_tu) / 10.0 * 1000.0

    tuned_preds = (tuned_probs >= 0.5).astype(int)
    tuned_macro_f1 = f1_score(y_test, tuned_preds, average="macro", zero_division=0)
    tuned_top1, tuned_cont = evaluate_top1_and_continuity(tuned_probs, meta_test, trip_arcs, y_test)
    tuned_ll = log_loss(y_test, tuned_probs)

    # 6. Save Best Model Artifact
    best_model_path = os.path.join(artifacts_dir, "best_tuned_model.joblib")
    joblib.dump(tuned_clf, best_model_path)
    print(f"Saved tuned model artifact: {best_model_path}")

    # 7. Comparison Table: Tuned vs Untuned (Deliverables Section 4)
    comparison_data = [
        {
            "Model Configuration": "Untuned Baseline (HistGradientBoosting)",
            "CV Macro F1-score": "N/A (Default)",
            "Student-testing Top-1 Accuracy": f"{untuned_top1 * 100.0:.2f}%",
            "Student-testing Macro F1-score": f"{untuned_macro_f1:.4f}",
            "Log Loss": f"{untuned_ll:.4f}",
            "Inference Time (ms / trip)": f"{untuned_infer_ms:.1f} ms",
            "Training Time": f"{untuned_fit_time:.2f} s"
        },
        {
            "Model Configuration": "Optuna Tuned (Best Trial)",
            "CV Macro F1-score": f"{best_value:.4f}",
            "Student-testing Top-1 Accuracy": f"{tuned_top1 * 100.0:.2f}%",
            "Student-testing Macro F1-score": f"{tuned_macro_f1:.4f}",
            "Log Loss": f"{tuned_ll:.4f}",
            "Inference Time (ms / trip)": f"{tuned_infer_ms:.1f} ms",
            "Training Time": f"{tuned_fit_time:.2f} s"
        }
    ]
    comp_df = pd.DataFrame(comparison_data)
    comp_csv = os.path.join(tables_dir, "tuned_vs_untuned_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nTuned vs Untuned Comparison saved: {comp_csv}")
    print(comp_df.to_string())

    # 8. Write Tuning Log
    log_file = os.path.join(logs_dir, "bayesian_optimization_log.md")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# PathSnap Bayesian Hyperparameter Optimisation Log\n\n")
        f.write(f"- **Search Strategy:** Optuna Tree-structured Parzen Estimator (TPE)\n")
        f.write(f"- **Objective:** Mean Observation-Level Top-1 Accuracy on 5-Fold GroupKFold\n")
        f.write(f"- **Trials Completed:** {len(trials_df)}\n")
        f.write(f"- **Best Trial Number:** {study.best_trial.number}\n")
        f.write(f"- **Best CV Top-1 Accuracy:** {best_value * 100.0:.2f}%\n")
        f.write(f"- **Total Search Duration:** {study_duration_s:.1f} seconds\n\n")
        f.write("## Best Hyperparameters\n\n```json\n" + json.dumps(best_params, indent=2) + "\n```\n\n")
        f.write("## Tuned vs Untuned Performance Comparison\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")
    print(f"Log written to: {log_file}")
    print("\nPHASE 6 BAYESIAN OPTIMIZATION COMPLETE!")


if __name__ == "__main__":
    main()
