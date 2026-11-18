"""
Multi-Model Experimentation and Benchmark Module for PathSnap.

Implements GroupKFold cross-validation across 6 distinct algorithm families:
1. Logistic Regression (Linear)
2. Gaussian Naive Bayes (Probabilistic)
3. Decision Tree Classifier (Tree-based)
4. Random Forest Classifier (Ensemble Bagging)
5. HistGradientBoosting Classifier (Ensemble Boosting)
6. Multi-Layer Perceptron (Neural Network)

Strictly adheres to:
- GroupKFold by trip_id (no trip split across folds)
- StandardScaler inside sklearn Pipeline to prevent data leakage
- Records all metrics required by Deliverables PDF Section 3.1
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

from src.features.feature_extractor import FEATURE_COLUMNS


def get_model_definitions() -> Dict[str, Dict[str, Any]]:
    """Returns the dictionary of the 6 model families with configurations."""
    models = {
        "LogisticRegression": {
            "family": "Linear",
            "key_config": "C=1.0, max_iter=1000, class_weight='balanced'",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42))
            ]),
            "has_loss_curve": False
        },
        "GaussianNB": {
            "family": "Probabilistic",
            "key_config": "var_smoothing=1e-8",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-8))
            ]),
            "has_loss_curve": False
        },
        "DecisionTree": {
            "family": "Tree-based",
            "key_config": "max_depth=10, min_samples_split=20, class_weight='balanced'",
            "pipeline": Pipeline([
                ("clf", DecisionTreeClassifier(max_depth=10, min_samples_split=20, class_weight="balanced", random_state=42))
            ]),
            "has_loss_curve": False
        },
        "RandomForest": {
            "family": "Ensemble (Bagging)",
            "key_config": "n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1",
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42))
            ]),
            "has_loss_curve": False
        },
        "HistGradientBoosting": {
            "family": "Ensemble (Boosting)",
            "key_config": "max_iter=120, learning_rate=0.08, max_depth=8, min_samples_leaf=15",
            "pipeline": Pipeline([
                ("clf", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=8, min_samples_leaf=15, random_state=42))
            ]),
            "has_loss_curve": True
        },
        "MLPClassifier": {
            "family": "Neural Network",
            "key_config": "hidden_layers=(64, 32), alpha=0.001, max_iter=100, early_stopping=True",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=100, early_stopping=True, random_state=42))
            ]),
            "has_loss_curve": True
        }
    }
    return models


def evaluate_top1_and_continuity(
    probs: np.ndarray,
    meta_df: pd.DataFrame,
    trip_arcs_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Tuple[float, float]:
    """
    Computes Observation-level Top-1 Accuracy and Route Continuity for a given set of predictions.
    - Top-1: For each (trip_id, obs_idx), pick candidate with highest probability; check if label == 1.
    - Continuity: For consecutive obs_idx in same trip, check if selected arcs are topologically connected.
    """
    meta_df = meta_df.copy()
    meta_df["prob"] = probs
    meta_df["is_true"] = y_true

    # Group by trip_id and obs_idx, get index of max prob
    idx_best = meta_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    best_df = meta_df.loc[idx_best].sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    # Top-1 accuracy
    top1_acc = float(np.mean(best_df["is_true"].values))

    # Continuity
    connected_steps = 0
    total_steps = 0

    for tid, grp in best_df.groupby("trip_id"):
        arcs = trip_arcs_dict.get(tid)
        if arcs is None:
            continue
        cands = grp["candidate_arc_id"].values
        for i in range(len(cands) - 1):
            total_steps += 1
            a1 = cands[i]
            a2 = cands[i + 1]
            if a1 == a2:
                connected_steps += 1
            else:
                to_node = arcs[a1, 1]
                from_node = arcs[a2, 0]
                if to_node == from_node:
                    connected_steps += 1

    continuity = (connected_steps / total_steps * 100.0) if total_steps > 0 else 0.0
    return top1_acc, continuity
