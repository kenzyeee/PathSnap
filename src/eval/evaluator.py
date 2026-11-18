"""
Server-Side Isolated Evaluation Module for PathSnap.

CRITICAL ARCHITECTURAL BARRIER:
This is the ONLY module in the entire project permitted to access and read the .route
ground-truth file during evaluation. The prediction pipeline (candidate generation,
feature extraction, model inference, HMM Viterbi decoding) has NO dependency on this module
and strictly operates without any ground-truth route inputs.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from src.data.loader import load_trip_data
from src.data.align_route import align_gps_to_route_dp


def evaluate_predictions_against_ground_truth(
    trip_id: str,
    trip_dir: str,
    predicted_point_arcs: List[int],
    reconstructed_route_arcs: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluates model and HMM predictions against ground truth .route on the server side.
    Returns:
        - accuracy: Pointwise Top-1 accuracy (%)
        - num_points: Total GPS observations
        - correct_points: Count of correctly predicted observations
        - route_precision: Precision of reconstructed route arcs vs ground-truth route arcs
        - route_recall: Recall of reconstructed route arcs vs ground-truth route arcs
        - route_f1: F1-score of reconstructed route arcs
        - route_jaccard: Jaccard similarity of reconstructed route arcs
    """
    route_file = os.path.join(trip_dir, f"{trip_id}.route")
    if not os.path.exists(route_file):
        return {
            "has_ground_truth": False,
            "message": "No server-side ground truth route available for this trip."
        }

    # Load trip and ground truth route
    trip = load_trip_data(trip_id, os.path.dirname(trip_dir), load_route=True)
    gt_route = trip.route
    if not gt_route:
        return {
            "has_ground_truth": False,
            "message": "Ground truth route file is empty."
        }

    # Derive server-side reference arc per observation
    alignment = align_gps_to_route_dp(trip)
    ref_arcs = alignment["reference_arcs"]

    N = min(len(predicted_point_arcs), len(ref_arcs))
    correct = sum(1 for i in range(N) if predicted_point_arcs[i] == ref_arcs[i])
    pointwise_acc = (correct / N * 100.0) if N > 0 else 0.0

    # Sequence-level metrics against ground-truth route
    pred_route_set = set(reconstructed_route_arcs) if reconstructed_route_arcs else set(predicted_point_arcs)
    gt_route_set = set(gt_route)

    intersection = len(pred_route_set.intersection(gt_route_set))
    union = len(pred_route_set.union(gt_route_set))
    jaccard = (intersection / union) if union > 0 else 0.0

    precision = (intersection / len(pred_route_set)) if pred_route_set else 0.0
    recall = (intersection / len(gt_route_set)) if gt_route_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "has_ground_truth": True,
        "trip_id": trip_id,
        "num_points": N,
        "correct_points": correct,
        "pointwise_accuracy_pct": round(pointwise_acc, 2),
        "route_precision": round(precision, 4),
        "route_recall": round(recall, 4),
        "route_f1": round(f1, 4),
        "route_jaccard": round(jaccard, 4),
        "gt_route_arc_count": len(gt_route),
        "pred_route_arc_count": len(pred_route_set)
    }
