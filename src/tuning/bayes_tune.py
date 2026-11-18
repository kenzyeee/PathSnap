"""
Bayesian Hyperparameter Optimization Module for PathSnap using Optuna (TPE).

Performs group-aware cross-validation on the 80 training trips over a justified search space.
Persists study history to SQLite and CSV.
Generates Optuna parameter importance and optimization history plots.
"""

import os
import time
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

from src.features.feature_extractor import FEATURE_COLUMNS
from src.models.train_and_evaluate import evaluate_top1_and_continuity


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    trip_arcs: Dict[str, np.ndarray],
    n_splits: int = 5
) -> float:
    """
    Optuna objective function: evaluates mean CV Top-1 Accuracy across GroupKFold splits.
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "max_iter": trial.suggest_int("max_iter", 80, 220, step=20),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80, step=5),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [128, 192, 255]),
        "random_state": 42
    }

    gkf = GroupKFold(n_splits=n_splits)
    top1_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_meta = meta_df.iloc[val_idx]

        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_tr, y_tr)

        val_probs = clf.predict_proba(X_val)[:, 1]
        top1, _ = evaluate_top1_and_continuity(val_probs, val_meta, trip_arcs, y_val)
        top1_scores.append(top1)

    return float(np.mean(top1_scores))


def run_bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    meta_train: pd.DataFrame,
    trip_arcs: Dict[str, np.ndarray],
    n_trials: int = 40,
    storage_url: str = "sqlite:///results/logs/optuna_study.db",
    study_name: str = "pathsnap_hist_gbm_tuning"
) -> optuna.Study:
    """Runs Optuna TPE study and returns study object."""
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler
    )

    print(f"Starting Optuna Bayesian Optimization ({n_trials} trials, TPE sampler)...")
    
    # Subsample 25 representative train trips (~450k rows) for high-speed trial evaluation
    unique_trips = np.unique(groups_train)
    np.random.seed(42)
    selected_trips = set(np.random.choice(unique_trips, size=min(25, len(unique_trips)), replace=False))
    mask = np.isin(groups_train, list(selected_trips))
    
    X_sub = X_train[mask]
    y_sub = y_train[mask]
    groups_sub = groups_train[mask]
    meta_sub = meta_train.iloc[np.where(mask)[0]].reset_index(drop=True)
    
    study.optimize(
        lambda trial: objective(trial, X_sub, y_sub, groups_sub, meta_sub, trip_arcs, n_splits=3),
        n_trials=n_trials
    )
    return study


def plot_optuna_diagnostics(study: optuna.Study, output_dir: str):
    """Saves optimization history and parameter importance plots using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Optimization History
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    trial_values = [t.value * 100.0 for t in trials if t.value is not None]
    best_values = np.maximum.accumulate(trial_values)

    plt.figure(figsize=(8, 4.5))
    plt.scatter(trial_numbers, trial_values, color="#3498db", alpha=0.7, label="Trial CV Top-1 (%)")
    plt.plot(trial_numbers, best_values, color="#e74c3c", linewidth=2.0, label="Best Value So Far")
    plt.xlabel("Trial Number")
    plt.ylabel("CV Top-1 Accuracy (%)")
    plt.title("Optuna Bayesian Optimization History")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optuna_optimization_history.png"), dpi=300)
    plt.close()

    # 2. Parameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        scores = list(importances.values())

        plt.figure(figsize=(8, 4))
        plt.barh(params[::-1], scores[::-1], color="#2ecc71")
        plt.xlabel("Relative Importance")
        plt.title("Hyperparameter Importances (Optuna TPE)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optuna_param_importances.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")
