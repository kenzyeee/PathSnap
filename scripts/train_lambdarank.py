"""
LambdaRank / LambdaMART Training and Lambda Optimization Script for PathSnap.

Executes:
1. Parallel feature extraction across training trips with observation candidate grouping.
2. Trains LightGBM LambdaRank model optimizing NDCG@1 ranking metric.
3. Post-training Lambda Optimization:
   Performs a systematic grid search over the scaling parameter lambda
   to find the optimal temperature multiplier that maximizes HMM sequence accuracy.
4. Generates:
   - artifacts/models/lambdarank_model.joblib
   - artifacts/models/best_tuned_model.joblib
   - results/plots/lambda_optimization_curve.png
   - results/plots/lambdarank_feature_importance.png
   - results/tables/lambda_optimization_results.csv
   - results/logs/lambdarank_training_log.md
5. Evaluates on the 10 student-test trips and verifies accuracy increase.
"""

import os
import sys
import time
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from joblib import Parallel, delayed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_trip_data
from src.data.align_route import align_gps_to_route_dp
from src.features.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_trip, FEATURE_COLUMNS
from src.models.lambdarank import LambdaRankModel
from src.hmm.viterbi import HMMMapMatcher
from src.hmm.route_reconstructor import reconstruct_continuous_route
from src.eval.evaluator import evaluate_predictions_against_ground_truth


def extract_trip_features_worker(tid: str, part_dir: str) -> Dict[str, Any]:
    """Worker function for parallel candidate generation and feature extraction."""
    trip = load_trip_data(tid, part_dir, load_route=True)
    alignment = align_gps_to_route_dp(trip)
    ref_arcs = alignment["reference_arcs"]
    
    cg = CandidateGenerator(
        trip,
        search_radius_m=65.0,
        sample_spacing_m=20.0,
        max_candidates=15,
        min_candidates=3
    )
    candidates = cg.generate_all_candidates()
    
    X_df, y, meta = extract_features_for_trip(trip, candidates, ref_arcs)
    
    # Compute group counts per observation
    group_counts = [len(candidates[t]) for t in range(len(trip.track)) if len(candidates[t]) > 0]
    
    return {
        "trip_id": tid,
        "X": X_df[FEATURE_COLUMNS].values,
        "y": y,
        "group": group_counts,
        "meta": meta,
        "candidates": candidates,
        "ref_arcs": ref_arcs,
        "num_obs": len(trip.track)
    }


def main():
    print("=" * 70)
    print("STARTING LAMBDARANK TRAINING & LAMBDA OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    train_dir = "data/Student folder/train"
    test_dir = "data/Student folder/student_test"
    artifacts_dir = "artifacts/models"
    plots_dir = "results/plots"
    tables_dir = "results/tables"
    logs_dir = "results/logs"
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    all_train_tids = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ])
    
    # Select 40 trips for training, 10 for validation
    train_tids = all_train_tids[:40]
    val_tids = all_train_tids[40:50]
    
    print(f"\n1. Extracting candidate features in parallel across {len(train_tids)} training trips...")
    t0 = time.time()
    train_results = Parallel(n_jobs=8)(
        delayed(extract_trip_features_worker)(tid, train_dir)
        for tid in train_tids
    )
    print(f"   Extraction complete in {time.time() - t0:.2f}s!")
    
    print(f"\n2. Extracting candidate features in parallel across {len(val_tids)} validation trips...")
    t1 = time.time()
    val_results = Parallel(n_jobs=8)(
        delayed(extract_trip_features_worker)(tid, train_dir)
        for tid in val_tids
    )
    print(f"   Extraction complete in {time.time() - t1:.2f}s!")
    
    # Aggregate training data
    X_train = np.vstack([res["X"] for res in train_results])
    y_train = np.concatenate([res["y"] for res in train_results])
    group_train = []
    for res in train_results:
        group_train.extend(res["group"])
    group_train = np.array(group_train, dtype=np.int32)
    
    # Aggregate validation data
    X_val = np.vstack([res["X"] for res in val_results])
    y_val = np.concatenate([res["y"] for res in val_results])
    group_val = []
    for res in val_results:
        group_val.extend(res["group"])
    group_val = np.array(group_val, dtype=np.int32)
    
    print(f"\nTraining set: {len(X_train):,} candidate rows across {len(group_train):,} observation queries")
    print(f"Validation set: {len(X_val):,} candidate rows across {len(group_val):,} observation queries")
    
    # 3. Train LambdaRank Model
    print("\n3. Training LightGBM LambdaRank (LambdaMART) Model...")
    t_train_start = time.time()
    lambda_model = LambdaRankModel(
        n_estimators=160,
        learning_rate=0.06,
        num_leaves=35,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        lambdarank_truncation_level=5,
        lambdarank_norm=True,
        random_state=42,
        n_jobs=-1
    )
    
    lambda_model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val]
    )
    train_time = time.time() - t_train_start
    print(f"   LambdaRank training finished in {train_time:.2f}s!")
    
    # Plot feature importances
    feat_imp = pd.Series(lambda_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind="barh", color="#2563EB")
    plt.title("PathSnap LambdaRank Feature Importances (Gain)", fontsize=14, fontweight="bold")
    plt.xlabel("Total Gain")
    plt.tight_layout()
    feat_imp_path = os.path.join(plots_dir, "lambdarank_feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"   Feature importance plot saved: {feat_imp_path}")
    
    # 4. Lambda Optimization (Grid Search over lambda scaling parameter)
    print("\n" + "=" * 70)
    print("4. OPTIMIZING LAMBDA (POST-TRAINING CALIBRATION & EMISSION TUNING)")
    print("=" * 70)
    print("Evaluating HMM sequence accuracy across validation trips for various lambda values...")
    
    lambda_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]
    lambda_eval_records = []
    
    # Pre-compute raw scores for validation trips to avoid redundant inference
    val_trip_scores = []
    for res in val_results:
        raw_scores = lambda_model.predict(res["X"])
        val_trip_scores.append(raw_scores)
        
    best_lambda = 1.0
    best_val_acc = -1.0
    
    for lam in lambda_grid:
        trip_accs = []
        trip_conts = []
        
        # Test calibrated HMM with this lambda
        hmm = HMMMapMatcher(
            sigma_z=5.0,
            beta=2.5,
            classifier_weight=3.5,
            max_graph_dist=200.0,
            large_gap_thresh_s=12.0
        )
        hmm.transition_model.unreachable_log_penalty = -50.0
        
        for idx, res in enumerate(val_results):
            raw_scores = val_trip_scores[idx]
            # Numerically stable calibrated probabilities with lambda
            scaled_scores = np.clip(lam * raw_scores, -50.0, 50.0)
            probs = 1.0 / (1.0 + np.exp(-scaled_scores))
            
            # Map probabilities to observations
            candidates = res["candidates"]
            ref_arcs = res["ref_arcs"]
            num_obs = res["num_obs"]
            
            cand_probs = []
            ptr = 0
            for t in range(num_obs):
                k = len(candidates[t])
                cand_probs.append(probs[ptr : ptr + k].tolist())
                ptr += k
                
            trip = load_trip_data(res["trip_id"], train_dir, load_route=False)
            pred_arcs = hmm.decode(trip, candidates, cand_probs)
            
            # Accuracy against ground truth
            correct = sum(1 for p, gt in zip(pred_arcs, ref_arcs) if p == gt)
            acc = (correct / num_obs * 100.0) if num_obs > 0 else 0.0
            trip_accs.append(acc)
            
        mean_acc = float(np.mean(trip_accs))
        lambda_eval_records.append({
            "lambda": lam,
            "val_pointwise_accuracy": round(mean_acc, 2),
            "min_accuracy": round(float(np.min(trip_accs)), 2),
            "max_accuracy": round(float(np.max(trip_accs)), 2)
        })
        print(f"   lambda = {lam:4.2f}  -->  Validation HMM Accuracy = {mean_acc:6.2f}%")
        
        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_lambda = lam

    print("-" * 70)
    print(f"OPTIMAL LAMBDA FOUND: lambda* = {best_lambda:.2f} (Validation Accuracy: {best_val_acc:.2f}%)")
    print("-" * 70)
    
    # Save Lambda Optimization Table
    lambda_df = pd.DataFrame(lambda_eval_records)
    lambda_csv_path = os.path.join(tables_dir, "lambda_optimization_results.csv")
    lambda_df.to_csv(lambda_csv_path, index=False)
    print(f"Saved lambda optimization results to: {lambda_csv_path}")
    
    # Plot Lambda Optimization Curve
    plt.figure(figsize=(9, 5))
    plt.plot(lambda_df["lambda"], lambda_df["val_pointwise_accuracy"], marker="o", color="#10B981", linewidth=2.5, markersize=7)
    plt.axvline(best_lambda, color="#EF4444", linestyle="--", label=f"Optimal $\\lambda^* = {best_lambda:.2f}$ (Acc: {best_val_acc:.2f}%)")
    plt.title("HMM Pointwise Accuracy vs. Lambda Scaling Factor $\\lambda$", fontsize=13, fontweight="bold")
    plt.xlabel("Scaling Parameter $\\lambda$ (Temperature Multiplier)", fontsize=11)
    plt.ylabel("Validation Pointwise Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    curve_path = os.path.join(plots_dir, "lambda_optimization_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved lambda optimization curve to: {curve_path}")
    
    # Set the optimized lambda on the model
    lambda_model.set_optimized_lambda(best_lambda)
    
    # Save LambdaRank model artifact
    lambdarank_model_path = os.path.join(artifacts_dir, "lambdarank_model.joblib")
    joblib.dump(lambda_model, lambdarank_model_path)
    print(f"\nTrained LambdaRank model serialized to: {lambdarank_model_path}")
    
    # 5. Evaluate on Student-Test Trips
    print("\n" + "=" * 70)
    print("5. EVALUATING LAMBDARANK ON 10 STUDENT-TEST TRIPS")
    print("=" * 70)
    
    test_tids = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.startswith(".")
    ])
    
    test_hmm = HMMMapMatcher(
        sigma_z=5.0,
        beta=2.5,
        classifier_weight=3.5,
        max_graph_dist=200.0,
        large_gap_thresh_s=12.0
    )
    test_hmm.transition_model.unreachable_log_penalty = -50.0
    
    student_records = []
    
    for idx, tid in enumerate(test_tids):
        t_start = time.time()
        # Strictly load without route
        trip = load_trip_data(tid, test_dir, load_route=False)
        
        cg = CandidateGenerator(
            trip,
            search_radius_m=65.0,
            sample_spacing_m=20.0,
            max_candidates=15,
            min_candidates=3
        )
        candidates = cg.generate_all_candidates()
        
        X_df, _, _ = extract_features_for_trip(trip, candidates, reference_arcs=None)
        
        # Predict using LambdaRank with optimized lambda
        probs = lambda_model.predict_proba(X_df[FEATURE_COLUMNS].values)[:, 1]
        
        cand_probs = []
        ptr = 0
        for t in range(len(trip.track)):
            k = len(candidates[t])
            cand_probs.append(probs[ptr : ptr + k].tolist())
            ptr += k
            
        # Top-1 candidate predictions (Classifier/Ranker only)
        top1_point_arcs = [
            candidates[t][int(np.argmax(cand_probs[t]))]["arc_id"]
            for t in range(len(trip.track))
        ]
        
        # HMM Viterbi Sequence Decoding
        hmm_point_arcs = test_hmm.decode(trip, candidates, cand_probs)
        
        # Continuous route reconstruction
        reconstructed = reconstruct_continuous_route(trip, hmm_point_arcs)
        elapsed_ms = (time.time() - t_start) * 1000.0
        
        # Isolated Ground Truth Evaluation
        eval_res = evaluate_predictions_against_ground_truth(
            trip_id=tid,
            trip_dir=os.path.join(test_dir, tid),
            predicted_point_arcs=hmm_point_arcs,
            reconstructed_route_arcs=reconstructed["route_arcs"]
        )
        
        clf_eval = evaluate_predictions_against_ground_truth(
            trip_id=tid,
            trip_dir=os.path.join(test_dir, tid),
            predicted_point_arcs=top1_point_arcs
        )
        
        clf_acc = clf_eval["pointwise_accuracy_pct"]
        hmm_acc = eval_res["pointwise_accuracy_pct"]
        gain = hmm_acc - clf_acc
        continuity = reconstructed["continuity_pct"]
        
        student_records.append({
            "trip_id": tid,
            "observations": len(trip.track),
            "clf_acc": clf_acc,
            "hmm_acc": hmm_acc,
            "gain": gain,
            "continuity": continuity,
            "route_f1": eval_res.get("route_f1", 0.0),
            "route_jaccard": eval_res.get("route_jaccard", 0.0),
            "latency_ms": round(elapsed_ms, 1)
        })
        
        print(f"  [{idx+1}/{len(test_tids)}] Trip {tid}: Ranker Acc={clf_acc:.2f}% -> HMM Acc={hmm_acc:.2f}% (gain: {gain:+.2f}%) | Cont={continuity:.1f}% | Latency={elapsed_ms:.1f}ms")
        
    mean_clf_acc = float(np.mean([r["clf_acc"] for r in student_records]))
    mean_hmm_acc = float(np.mean([r["hmm_acc"] for r in student_records]))
    mean_cont = float(np.mean([r["continuity"] for r in student_records]))
    mean_f1 = float(np.mean([r["route_f1"] for r in student_records]))
    mean_jaccard = float(np.mean([r["route_jaccard"] for r in student_records]))
    mean_latency = float(np.mean([r["latency_ms"] for r in student_records]))
    
    print("\n" + "=" * 70)
    print("OVERALL STUDENT-TEST RESULTS (WITH LAMBDARANK):")
    print(f"  Mean Ranker-Only Top-1 Accuracy:  {mean_clf_acc:.2f}%")
    print(f"  Mean LambdaRank + HMM Accuracy:    {mean_hmm_acc:.2f}%")
    print(f"  Mean Route Continuity:             {mean_cont:.2f}%")
    print(f"  Mean Route F1-Score:               {mean_f1:.4f}")
    print(f"  Mean Route Jaccard:                {mean_jaccard:.4f}")
    print(f"  Mean Latency per Trip:             {mean_latency:.1f} ms")
    print("=" * 70)
    
    # Save student test evaluation table
    eval_df = pd.DataFrame(student_records)
    eval_csv_path = os.path.join(tables_dir, "lambdarank_student_test_eval.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Saved student test evaluation table to: {eval_csv_path}")
    
    # If LambdaRank achieved higher accuracy than previous best (92.70%), set as best_tuned_model.joblib
    if mean_hmm_acc >= 92.70:
        best_model_path = os.path.join(artifacts_dir, "best_tuned_model.joblib")
        joblib.dump(lambda_model, best_model_path)
        print(f"--> NEW BEST MODEL DEPLOYED: {best_model_path} (Accuracy: {mean_hmm_acc:.2f}%)")
    
    # Write Markdown Log
    log_content = f"""# PathSnap LambdaRank & Lambda Optimization Report

## 1. Executive Summary
- **Algorithm**: LightGBM LambdaRank / LambdaMART (`LGBMRanker`)
- **Objective Function**: `lambdarank` (Direct optimization of pairwise/listwise NDCG@1)
- **Features Used**: 18 geometric, kinematic, and topological features
- **Training Set**: {len(X_train):,} candidate rows across {len(group_train):,} observation queries
- **Optimized Lambda Parameter**: **$\\lambda^* = {best_lambda:.2f}$**
- **Student-Test HMM Accuracy**: **{mean_hmm_acc:.2f}%**
- **Student-Test Continuity**: **{mean_cont:.2f}%**
- **Mean Latency per Trip**: **{mean_latency:.1f} ms**

## 2. Lambda Optimization Results
| Lambda ($\\lambda$) | Validation HMM Accuracy (%) | Min Trip Accuracy (%) | Max Trip Accuracy (%) |
|---|---|---|---|
"""
    for r in lambda_eval_records:
        marker = " **(OPTIMAL)**" if r["lambda"] == best_lambda else ""
        log_content += f"| {r['lambda']:.2f}{marker} | {r['val_pointwise_accuracy']:.2f}% | {r['min_accuracy']:.2f}% | {r['max_accuracy']:.2f}% |\n"

    log_content += f"""
## 3. Student-Test Evaluation (10 Trips)
| Trip ID | Observations | Ranker Top-1 Acc (%) | LambdaRank + HMM Acc (%) | Gain (%) | Continuity (%) | Route F1 | Latency (ms) |
|---|---|---|---|---|---|---|---|
"""
    for r in student_records:
        log_content += f"| {r['trip_id']} | {r['observations']} | {r['clf_acc']:.2f}% | {r['hmm_acc']:.2f}% | {r['gain']:+.2f}% | {r['continuity']:.1f}% | {r['route_f1']:.4f} | {r['latency_ms']:.1f} |\n"

    log_content += f"""
**Summary Averages**:
- Mean Ranker Top-1 Accuracy: **{mean_clf_acc:.2f}%**
- Mean LambdaRank + HMM Accuracy: **{mean_hmm_acc:.2f}%**
- Mean Route Continuity: **{mean_cont:.2f}%**
- Mean F1-Score: **{mean_f1:.4f}**
- Mean Latency: **{mean_latency:.1f} ms**
"""
    log_path = os.path.join(logs_dir, "lambdarank_training_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"Saved comprehensive log to: {log_path}")


if __name__ == "__main__":
    main()
