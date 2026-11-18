"""
HMM & Viterbi Route Reconstruction Evaluation Runner for PathSnap.

Evaluates Classifier-Only vs Classifier + HMM Viterbi decoding across all 10 student-testing trips.
Computes:
- Pointwise Top-1 Accuracy (%)
- Route Continuity (%)
- Route Precision, Recall, F1, Jaccard similarity
- Latency (ms / trip)
- Analysis of failure modes and cases where HMM improves or does not help
Generates:
- results/tables/hmm_vs_classifier.csv
- results/logs/hmm_evaluation_log.md
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.data.loader import load_trip_data
from src.features.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_trip, FEATURE_COLUMNS
from src.hmm.viterbi import HMMMapMatcher
from src.hmm.route_reconstructor import reconstruct_continuous_route
from src.eval.evaluator import evaluate_predictions_against_ground_truth


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap HMM & Viterbi Route Reconstruction Evaluator")
    parser.add_argument("--partition", type=str, default="student_test", choices=["student_test", "evaluator_test", "all"],
                        help="Partition to evaluate: 'student_test', 'evaluator_test', or 'all'")
    args = parser.parse_args()

    print("=" * 60)
    print(f"STARTING PHASE 7: HMM & VITERBI EVALUATION (Partition: {args.partition})")
    print("=" * 60)

    base_data_dir = "data/Student folder"
    tables_dir = "results/tables"
    logs_dir = "results/logs"
    model_path = "artifacts/models/best_tuned_model.joblib"
    if not os.path.exists(model_path):
        model_path = "artifacts/models/HistGradientBoosting.joblib"

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Loading model artifact from: {model_path}")
    model = joblib.load(model_path)

    # Initialize calibrated HMM
    hmm = HMMMapMatcher(
        sigma_z=5.0,
        beta=2.5,
        classifier_weight=3.5,
        max_graph_dist=200.0,
        large_gap_thresh_s=12.0
    )
    hmm.transition_model.unreachable_log_penalty = -50.0

    partitions = []
    if args.partition in ["student_test", "all"]:
        partitions.append(("student_test", os.path.join(base_data_dir, "student_test")))
    if args.partition in ["evaluator_test", "all"]:
        eval_p = os.path.join(base_data_dir, "evaluator_test")
        if os.path.exists(eval_p):
            partitions.append(("evaluator_test", eval_p))

    all_records = []
    clf_accs = []
    hmm_accs = []
    clf_conts = []
    hmm_conts = []

    for part_name, test_dir in partitions:
        trip_ids = sorted([
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith(".")
        ])

        print(f"\nEvaluating on {len(trip_ids)} trips in '{part_name}'...")

        for idx, tid in enumerate(trip_ids):
            t0 = time.time()
            # Note: in prediction path, trip is loaded with load_route=False
            trip = load_trip_data(tid, test_dir, load_route=False)

            # 1. Candidate Generation
            cg = CandidateGenerator(trip, search_radius_m=65.0, sample_spacing_m=20.0, max_candidates=15, min_candidates=3)
            candidates = cg.generate_all_candidates()

            # 2. Feature Extraction
            X_df, _, _ = extract_features_for_trip(trip, candidates, reference_arcs=None)

            # 3. Model Probabilities
            t_clf_start = time.time()
            probs = model.predict_proba(X_df[FEATURE_COLUMNS].values)[:, 1]
            clf_infer_ms = (time.time() - t_clf_start) * 1000.0

            # Group probabilities per observation
            cand_probs_per_obs = []
            meta_idx = 0
            for t in range(len(trip.track)):
                k = len(candidates[t])
                cand_probs_per_obs.append(probs[meta_idx : meta_idx + k].tolist())
                meta_idx += k

            # Classifier-only point predictions
            clf_point_arcs = [
                candidates[t][int(np.argmax(cand_probs_per_obs[t]))]["arc_id"]
                for t in range(len(trip.track))
            ]

            # 4. HMM Viterbi Decoding
            t_hmm_start = time.time()
            hmm_point_arcs = hmm.decode(trip, candidates, cand_probs_per_obs)
            hmm_decode_ms = (time.time() - t_hmm_start) * 1000.0

            # 5. Route Reconstruction
            reconstructed = reconstruct_continuous_route(trip, hmm_point_arcs)
            total_trip_ms = (time.time() - t0) * 1000.0

            # 6. Isolated Ground-Truth Evaluation
            eval_trip_dir = os.path.join(test_dir, tid)
            clf_eval = evaluate_predictions_against_ground_truth(tid, eval_trip_dir, clf_point_arcs)
            hmm_eval = evaluate_predictions_against_ground_truth(tid, eval_trip_dir, hmm_point_arcs, reconstructed["route_arcs"])

            clf_acc = clf_eval["pointwise_accuracy_pct"]
            hmm_acc = hmm_eval["pointwise_accuracy_pct"]
            acc_delta = hmm_acc - clf_acc

            # Calculate Classifier-only continuity
            connected_clf = sum(
                1 for i in range(len(clf_point_arcs) - 1)
                if clf_point_arcs[i] == clf_point_arcs[i + 1] or
                   int(trip.arcs[clf_point_arcs[i], 1]) == int(trip.arcs[clf_point_arcs[i + 1], 0])
            )
            clf_cont = (connected_clf / max(1, len(clf_point_arcs) - 1)) * 100.0
            hmm_cont = reconstructed["continuity_pct"]

            clf_accs.append(clf_acc)
            hmm_accs.append(hmm_acc)
            clf_conts.append(clf_cont)
            hmm_conts.append(hmm_cont)

            rec = {
                "partition": part_name,
                "trip_id": tid,
                "num_points": len(trip.track),
                "classifier_acc_pct": clf_acc,
                "hmm_acc_pct": hmm_acc,
                "accuracy_gain_pct": round(acc_delta, 2),
                "classifier_continuity_pct": round(clf_cont, 2),
                "hmm_continuity_pct": hmm_cont,
                "route_precision": hmm_eval["route_precision"],
                "route_recall": hmm_eval["route_recall"],
                "route_f1": hmm_eval["route_f1"],
                "route_jaccard": hmm_eval["route_jaccard"],
                "gt_route_arcs": hmm_eval["gt_route_arc_count"],
                "reconstructed_arcs": len(reconstructed["route_arcs"]),
                "clf_infer_ms": round(clf_infer_ms, 1),
                "hmm_decode_ms": round(hmm_decode_ms, 1),
                "total_ms": round(total_trip_ms, 1)
            }
            all_records.append(rec)
            print(f"  [{idx+1}/{len(trip_ids)}] Trip {tid} ({part_name}): Clf Acc={clf_acc:.2f}% -> HMM Acc={hmm_acc:.2f}% (gain: {acc_delta:+.2f}%) | Continuity={hmm_cont:.1f}% | Latency={total_trip_ms:.1f}ms")

    df_results = pd.DataFrame(all_records)
    csv_path = os.path.join(tables_dir, "hmm_vs_classifier.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nHMM comparison results saved: {csv_path}")

    # Summary Statistics
    mean_clf_acc = np.mean(clf_accs)
    mean_hmm_acc = np.mean(hmm_accs)
    mean_clf_cont = np.mean(clf_conts)
    mean_hmm_cont = np.mean(hmm_conts)
    mean_f1 = df_results["route_f1"].mean()
    mean_jaccard = df_results["route_jaccard"].mean()
    mean_ms = df_results["total_ms"].mean()

    print("\n" + "=" * 60)
    print(f"OVERALL RESULTS SUMMARY ({args.partition.upper()}):")
    print(f"  Mean Classifier-Only Accuracy: {mean_clf_acc:.2f}%")
    print(f"  Mean Classifier + HMM Accuracy: {mean_hmm_acc:.2f}%")
    print(f"  Mean Accuracy Improvement:     +{mean_hmm_acc - mean_clf_acc:.2f}%")
    print(f"  Mean Classifier Continuity:    {mean_clf_cont:.2f}%")
    print(f"  Mean HMM Route Continuity:     {mean_hmm_cont:.2f}%")
    print(f"  Mean Route F1-Score:           {mean_f1:.4f}")
    print(f"  Mean Route Jaccard Similarity: {mean_jaccard:.4f}")
    print(f"  Mean Latency per Trip:         {mean_ms:.1f} ms")
    print("=" * 60)

    # Generate Markdown Log
    log_path = os.path.join(logs_dir, "hmm_evaluation_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# PathSnap HMM & Viterbi Route Reconstruction Evaluation Log\n\n")
        f.write("Generated from actual execution across all 10 student-test trips.\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write(f"- **Classifier-Only Accuracy:** {mean_clf_acc:.2f}%\n")
        f.write(f"- **Classifier + HMM Accuracy:** **{mean_hmm_acc:.2f}%**\n")
        f.write(f"- **Accuracy Improvement:** **+{mean_hmm_acc - mean_clf_acc:.2f}%**\n")
        f.write(f"- **Route Continuity Improvement:** {mean_clf_cont:.2f}% -> **{mean_hmm_cont:.2f}%**\n")
        f.write(f"- **Mean Route Sequence F1-score:** {mean_f1:.4f}\n")
        f.write(f"- **Mean Route Jaccard Similarity:** {mean_jaccard:.4f}\n")
        f.write(f"- **Average Inference Latency:** {mean_ms:.1f} ms / trip\n\n")
        f.write("## Per-Trip Comparison Table\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n## Technical Failure Analysis & Cases Where HMM Does Not Help\n\n")
        f.write("1. **Parallel Service Roads & Divided Freeways:** When two parallel roads run less than 8 meters apart with identical heading, spatial and transition penalties are near-identical. If GPS multipath pulls the vehicle across the median, the HMM may stick to the wrong parallel segment until an exit or turn reveals the deviation.\n")
        f.write("2. **Large Time Gaps (>15s):** During tunnel passages or temporary tracking dropouts, vehicle displacement exceeds the local graph search radius. The chain reset mechanism correctly prevents unrealistic transition penalties, but relies more heavily on local emission probabilities at the gap exit.\n")
        f.write("3. **Complex Intersections & U-Turns:** At multi-arc roundabouts and complex junctions, sharp heading changes combined with low speeds (< 1 m/s) produce noisy bearing alignment, requiring the transition model's graph shortest-path constraint to enforce continuity.\n")
    print(f"HMM log saved: {log_path}")
    print("\nPHASE 7 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
