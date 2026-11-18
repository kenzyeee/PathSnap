"""
Multi-Model Benchmark & Cross-Validation Execution Script for PathSnap.

Runs 5-fold GroupKFold cross-validation across 6 distinct model families:
1. Logistic Regression (Linear)
2. Gaussian Naive Bayes (Probabilistic)
3. Decision Tree (Tree-based)
4. Random Forest (Ensemble Bagging)
5. HistGradientBoosting (Ensemble Boosting)
6. Multi-Layer Perceptron (Neural Network)

Generates:
- results/tables/model_comparison.csv (Matches Deliverables PDF Section 3.1)
- results/tables/cv_fold_metrics.csv
- results/plots/ (All bar charts, loss curves, confusion matrices)
- artifacts/models/ (Serialized trained model pipelines)
- results/logs/model_experimentation_log.md
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    log_loss, confusion_matrix
)
import joblib

from src.features.feature_extractor import FEATURE_COLUMNS
from src.models.train_and_evaluate import get_model_definitions, evaluate_top1_and_continuity
from src.models.model_visualizer import (
    plot_model_comparison_bars,
    plot_confusion_matrices,
    plot_loss_curves
)
from src.data.loader import load_trip_data


def load_all_trip_arcs(data_dir: str) -> Dict[str, np.ndarray]:
    """Preloads arcs array for all trips in train and student_test for continuity checks."""
    arcs_dict = {}
    for partition in ["train", "student_test"]:
        part_dir = os.path.join(data_dir, partition)
        if not os.path.exists(part_dir):
            continue
        for tid in os.listdir(part_dir):
            arcs_path = os.path.join(part_dir, tid, f"{tid}.arcs")
            if os.path.exists(arcs_path):
                with open(arcs_path) as f:
                    arcs_dict[tid] = np.array([list(map(int, line.strip().split())) for line in f])
    return arcs_dict


def run_benchmark():
    print("=" * 60)
    print("STARTING PHASE 5: MULTI-MODEL BENCHMARK EXPERIMENTS")
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

    # 1. Load Processed Datasets
    print("\n1. Loading processed training and student-testing feature datasets...")
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_dataset.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "student_test_dataset.parquet"))

    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train_full = train_df["label"].values
    groups_train = train_df["trip_id"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values
    meta_test = test_df[["trip_id", "obs_idx", "candidate_arc_id"]].copy()

    print(f"Train samples: {len(X_train_full)} ({train_df['trip_id'].nunique()} trips)")
    print(f"Student-Test samples: {len(X_test)} ({test_df['trip_id'].nunique()} trips)")

    # Preload arcs for continuity evaluation
    trip_arcs = load_all_trip_arcs(data_dir)

    models_dict = get_model_definitions()
    gkf = GroupKFold(n_splits=5)

    comparison_records = []
    cv_fold_records = []
    confusion_matrices = {}
    loss_curves = {}

    # 2. Iterate through each model family
    for model_name, m_info in models_dict.items():
        print(f"\nEvaluating: {model_name} ({m_info['family']})...")
        pipeline = m_info["pipeline"]
        key_config = m_info["key_config"]

        fold_macro_f1s = []
        fold_top1_accs = []
        fold_continuities = []
        fold_log_losses = []

        # 5-Fold GroupKFold Cross-Validation on 80 Train Trips
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups=groups_train)):
            X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
            val_meta = train_df.iloc[val_idx][["trip_id", "obs_idx", "candidate_arc_id"]].copy()

            # Clone and fit
            from sklearn.base import clone
            fold_pipe = clone(pipeline)
            fold_pipe.fit(X_tr, y_tr)

            # Predict probabilities
            val_probs = fold_pipe.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)

            f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            ll = log_loss(y_val, val_probs)
            top1, cont = evaluate_top1_and_continuity(val_probs, val_meta, trip_arcs, y_val)

            fold_macro_f1s.append(f1)
            fold_top1_accs.append(top1)
            fold_continuities.append(cont)
            fold_log_losses.append(ll)

            cv_fold_records.append({
                "model_name": model_name,
                "fold": fold_idx,
                "macro_f1": round(f1, 4),
                "top1_acc": round(top1, 4),
                "continuity": round(cont, 2),
                "log_loss": round(ll, 4)
            })

        cv_mean_f1 = np.mean(fold_macro_f1s)
        cv_std_f1 = np.std(fold_macro_f1s)
        print(f"  CV 5-Fold Macro F1: {cv_mean_f1:.4f} ± {cv_std_f1:.4f}")
        print(f"  CV Mean Top-1 Acc: {np.mean(fold_top1_accs)*100.0:.2f}% | Continuity: {np.mean(fold_continuities):.2f}%")

        # Refit on full 80 train trips
        t_fit_start = time.time()
        pipeline.fit(X_train_full, y_train_full)
        train_time_s = time.time() - t_fit_start

        # Measure Train metrics to check for overfitting
        train_probs_sample = pipeline.predict_proba(X_train_full[:20000])[:, 1]
        train_preds_sample = (train_probs_sample >= 0.5).astype(int)
        train_macro_f1 = f1_score(y_train_full[:20000], train_preds_sample, average="macro", zero_division=0)

        # Measure Student-Test inference time and performance
        t_infer_start = time.time()
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        total_infer_time_s = time.time() - t_infer_start
        ms_per_trip = (total_infer_time_s / 10.0) * 1000.0

        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        test_acc = accuracy_score(y_test, test_preds)
        test_log_loss = log_loss(y_test, test_probs)
        test_top1_acc, test_continuity = evaluate_top1_and_continuity(test_probs, meta_test, trip_arcs, y_test)

        cm = confusion_matrix(y_test, test_preds)
        confusion_matrices[model_name] = cm

        # Extract loss curve if model produces epoch history
        if m_info["has_loss_curve"]:
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "loss_curve_"):
                loss_curves[model_name] = list(clf.loss_curve_)
            elif hasattr(clf, "train_score_"):
                loss_curves[model_name] = list(clf.train_score_)

        # Save trained pipeline artifact
        artifact_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
        joblib.dump(pipeline, artifact_path)

        rec = {
            "model_name": model_name,
            "algorithm_family": m_info["family"],
            "key_configuration": key_config,
            "cv_macro_f1_mean": round(cv_mean_f1, 4),
            "cv_macro_f1_std": round(cv_std_f1, 4),
            "cv_macro_f1_str": f"{cv_mean_f1:.4f} ± {cv_std_f1:.4f}",
            "student_test_top1_acc": round(test_top1_acc, 4),
            "student_test_continuity": round(test_continuity, 2),
            "student_test_macro_f1": round(test_macro_f1, 4),
            "student_test_precision": round(test_precision, 4),
            "student_test_recall": round(test_recall, 4),
            "student_test_accuracy": round(test_acc, 4),
            "student_test_log_loss": round(test_log_loss, 4),
            "train_macro_f1": round(train_macro_f1, 4),
            "train_vs_cv_gap": round(train_macro_f1 - cv_mean_f1, 4),
            "train_time_s": round(train_time_s, 2),
            "inference_time_ms_per_trip": round(ms_per_trip, 2)
        }
        comparison_records.append(rec)
        print(f"  Test Top-1 Acc: {test_top1_acc*100.0:.2f}% | Test Macro F1: {test_macro_f1:.4f} | Latency: {ms_per_trip:.1f}ms/trip")

    # 3. Save Summary Comparison Tables
    results_df = pd.DataFrame(comparison_records)
    cv_df = pd.DataFrame(cv_fold_records)

    # Table matching Deliverables PDF Section 3.1
    deliv_table = results_df[[
        "model_name", "algorithm_family", "key_configuration",
        "cv_macro_f1_str", "student_test_top1_acc", "student_test_continuity",
        "student_test_macro_f1", "student_test_log_loss", "inference_time_ms_per_trip"
    ]].rename(columns={
        "model_name": "Model",
        "algorithm_family": "Algorithm Family",
        "key_configuration": "Key Configuration",
        "cv_macro_f1_str": "CV Macro F1-score",
        "student_test_top1_acc": "Top-1 Accuracy",
        "student_test_continuity": "Route continuity",
        "student_test_macro_f1": "Student-Test Macro F1-score",
        "student_test_log_loss": "Log Loss",
        "inference_time_ms_per_trip": "Inference Time (ms)"
    })

    deliv_table_path = os.path.join(tables_dir, "model_comparison.csv")
    deliv_table.to_csv(deliv_table_path, index=False)
    print(f"\nDeliverables Table 3.1 saved: {deliv_table_path}")
    print(deliv_table.to_string())

    full_results_path = os.path.join(tables_dir, "model_benchmark_full.csv")
    results_df.to_csv(full_results_path, index=False)
    cv_df.to_csv(os.path.join(tables_dir, "cv_fold_metrics.csv"), index=False)

    # 4. Generate Visualizations
    print("\n4. Generating benchmark plots...")
    plot_model_comparison_bars(results_df, plots_dir)
    plot_confusion_matrices(confusion_matrices, plots_dir)
    plot_loss_curves(loss_curves, plots_dir)
    print("All plots generated in results/plots/")

    # 5. Generate Experimentation Log
    log_path = os.path.join(logs_dir, "model_experimentation_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# PathSnap Model Experimentation and Benchmark Log\n\n")
        f.write("Generated from real execution of 5-Fold GroupKFold cross-validation.\n\n")
        f.write("## Consolidated Model Comparison (Deliverables Section 3.1)\n\n")
        f.write(deliv_table.to_markdown(index=False))
        f.write("\n\n## Per-Model Details\n\n")
        for rec in comparison_records:
            f.write(f"### {rec['model_name']} ({rec['algorithm_family']})\n")
            f.write(f"- **Key Config:** `{rec['key_configuration']}`\n")
            f.write(f"- **CV Macro F1:** {rec['cv_macro_f1_str']}\n")
            f.write(f"- **Student-Test Top-1 Accuracy:** {rec['student_test_top1_acc']*100.0:.2f}%\n")
            f.write(f"- **Student-Test Macro F1:** {rec['student_test_macro_f1']:.4f}\n")
            f.write(f"- **Student-Test Log Loss:** {rec['student_test_log_loss']:.4f}\n")
            f.write(f"- **Route Continuity:** {rec['student_test_continuity']:.2f}%\n")
            f.write(f"- **Training Time:** {rec['train_time_s']:.2f}s | **Inference Latency:** {rec['inference_time_ms_per_trip']:.1f}ms/trip\n")
            f.write(f"- **Overfitting Gap (Train vs CV Macro F1):** {rec['train_vs_cv_gap']:+.4f}\n\n")
            
    print(f"Experimentation log saved: {log_path}")
    print("\nPHASE 5 BENCHMARK COMPLETE!")


if __name__ == "__main__":
    run_benchmark()
