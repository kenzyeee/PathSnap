"""
Model Visualization Module for PathSnap.

Generates mandatory Deliverables plots (Deliverables PDF Section 3.3):
1. Cross-validation Macro F1-score comparison across all models
2. Student-testing Top-1 Accuracy and Macro F1-score comparison
3. Log Loss comparison across models
4. Computational efficiency (Inference Time & Training Time)
5. Confusion matrices for all evaluated models
6. Training/validation loss curves for iterative models (MLP, HistGradientBoosting)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List


def plot_model_comparison_bars(results_df: pd.DataFrame, output_dir: str):
    """Generates comparison bar charts for CV Macro F1, Student Test metrics, Log Loss, and Latency."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. CV Macro F1 vs Student-Test Macro F1
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.35
    
    cv_means = results_df["cv_macro_f1_mean"].values
    cv_stds = results_df["cv_macro_f1_std"].values
    test_f1 = results_df["student_test_macro_f1"].values
    
    plt.bar(x - width/2, cv_means, width, yerr=cv_stds, capsize=4, label="CV Macro F1 (5-Fold)", color="#1f77b4")
    plt.bar(x + width/2, test_f1, width, label="Student-Test Macro F1", color="#2ca02c")
    
    plt.ylabel("Macro F1-Score")
    plt.title("Model Comparison: Cross-Validation vs Student-Test Macro F1")
    plt.xticks(x, results_df["model_name"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "macro_f1_comparison.png"), dpi=300)
    plt.close()

    # 2. Student-Test Top-1 Accuracy and Route Continuity
    plt.figure(figsize=(10, 5))
    top1_acc = results_df["student_test_top1_acc"].values * 100.0
    continuity = results_df["student_test_continuity"].values
    
    plt.bar(x - width/2, top1_acc, width, label="Student-Test Top-1 Accuracy (%)", color="#ff7f0e")
    plt.bar(x + width/2, continuity, width, label="Route Continuity (%)", color="#9467bd")
    
    plt.ylabel("Percentage (%)")
    plt.title("Model Comparison: Observation Top-1 Accuracy & Route Continuity")
    plt.xticks(x, results_df["model_name"], rotation=20, ha="right")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top1_accuracy_and_continuity.png"), dpi=300)
    plt.close()

    # 3. Log Loss Comparison
    plt.figure(figsize=(9, 4.5))
    log_losses = results_df["student_test_log_loss"].values
    colors = sns.color_palette("rocket", len(results_df))
    plt.bar(results_df["model_name"], log_losses, color=colors, width=0.55)
    plt.ylabel("Log Loss (Cross-Entropy)")
    plt.title("Probability Calibration: Student-Test Log Loss Across Models")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "log_loss_comparison.png"), dpi=300)
    plt.close()

    # 4. Computational Efficiency (Training and Inference Time)
    plt.figure(figsize=(10, 4.5))
    train_times = results_df["train_time_s"].values
    infer_times = results_df["inference_time_ms_per_trip"].values
    
    plt.subplot(1, 2, 1)
    plt.bar(results_df["model_name"], train_times, color="#34495e", width=0.55)
    plt.ylabel("Training Time (seconds)")
    plt.title("Model Training Time")
    plt.xticks(rotation=25, ha="right")
    
    plt.subplot(1, 2, 2)
    plt.bar(results_df["model_name"], infer_times, color="#e67e22", width=0.55)
    plt.ylabel("Inference Latency (ms / trip)")
    plt.title("Inference Latency per Trip")
    plt.xticks(rotation=25, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "computational_efficiency.png"), dpi=300)
    plt.close()


def plot_confusion_matrices(conf_matrices: Dict[str, np.ndarray], output_dir: str):
    """Plots confusion matrix heatmap for each evaluated model."""
    os.makedirs(output_dir, exist_ok=True)
    for model_name, cm in conf_matrices.items():
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative (0)", "Positive (1)"],
            yticklabels=["Negative (0)", "Positive (1)"]
        )
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Candidate")
        plt.ylabel("True Reference Arc")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
        plt.close()


def plot_loss_curves(loss_histories: Dict[str, List[float]], output_dir: str):
    """
    Plots real iteration/epoch loss curves ONLY for models that produce training histories.
    Strictly adheres to: 'Do not create artificial training curves for algorithms that do not produce training histories.'
    """
    os.makedirs(output_dir, exist_ok=True)
    for model_name, losses in loss_histories.items():
        if not losses:
            continue
        plt.figure(figsize=(6.5, 4))
        plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, color="#d9534f", linewidth=1.5)
        plt.title(f"Training Loss History: {model_name}")
        plt.xlabel("Iteration / Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"loss_curve_{model_name}.png"), dpi=300)
        plt.close()
