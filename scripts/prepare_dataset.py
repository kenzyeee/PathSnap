"""
Data Preparation and Feature Generation Pipeline for PathSnap.

Extracts features and candidate labels for all 80 training trips and 10 student-testing trips.
Computes and reports Candidate Miss Rate (CMR) across all trips.
Saves processed feature datasets and alignment records to data/processed/.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_trip_data, TripData
from src.data.align_route import align_gps_to_route_dp
from src.features.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_trip, FEATURE_COLUMNS


def process_partition(
    partition: str,
    base_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """Processes all trips in a partition, generates candidates, extracts features, and saves dataset."""
    part_dir = os.path.join(base_dir, partition)
    trip_ids = sorted([
        d for d in os.listdir(part_dir)
        if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")
    ])
    
    print(f"\nProcessing partition: '{partition}' ({len(trip_ids)} trips)...")
    
    all_X = []
    all_y = []
    all_meta = []
    
    total_pts = 0
    total_misses = 0
    trip_cmr_dict = {}
    alignment_stats = []
    
    t_start = time.time()
    for idx, tid in enumerate(trip_ids):
        t0 = time.time()
        # In train, load_route=True is used to derive training labels.
        # In student_test, load_route=True is used strictly for offline validation/reporting.
        trip = load_trip_data(tid, part_dir, load_route=True)
        
        # Monotonic Ground-Truth Alignment
        alignment = align_gps_to_route_dp(trip)
        ref_arcs = alignment["reference_arcs"]
        alignment_stats.append({
            "trip_id": tid,
            "mean_perp_dist": alignment["mean_perp_dist"],
            "median_perp_dist": alignment["median_perp_dist"],
            "outlier_pct": alignment["outlier_pct"]
        })
        
        # Multi-Point Candidate Generation
        cg = CandidateGenerator(trip, search_radius_m=65.0, sample_spacing_m=35.0, max_candidates=15, min_candidates=3)
        candidates = cg.generate_all_candidates()
        
        # CMR verification
        N = len(trip.track)
        misses = 0
        for t in range(N):
            cand_arc_ids = [c["arc_id"] for c in candidates[t]]
            if ref_arcs[t] not in cand_arc_ids:
                misses += 1
                
        total_pts += N
        total_misses += misses
        cmr = (misses / N * 100.0) if N > 0 else 0.0
        trip_cmr_dict[tid] = cmr
        
        # Feature Extraction
        X_df, y, meta = extract_features_for_trip(trip, candidates, ref_arcs)
        all_X.append(X_df)
        all_y.append(y)
        all_meta.extend(meta)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(trip_ids)}] Processed {tid} | CMR={cmr:.2f}% | Elapsed: {elapsed:.1f}s")
            
    # Concatenate all
    full_X = pd.concat(all_X, ignore_index=True)
    full_y = np.concatenate(all_y)
    meta_df = pd.DataFrame(all_meta)
    
    overall_cmr = (total_misses / total_pts * 100.0) if total_pts > 0 else 0.0
    print(f"\nPartition '{partition}' Summary:")
    print(f"  Total GPS observations: {total_pts}")
  
    print(f"  Total candidate rows: {len(full_X)}")
    print(f"  Positive labels (y=1): {full_y.sum()} ({full_y.sum() / len(full_y) * 100.0:.2f}%)")
    print(f"  Overall Candidate Miss Rate (CMR): {overall_cmr:.2f}% ({total_misses}/{total_pts})")
    print(f"  Total time: {time.time() - t_start:.2f}s")
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    dataset_df = full_X.copy()
    dataset_df["label"] = full_y
    dataset_df["trip_id"] = meta_df["trip_id"]
    dataset_df["obs_idx"] = meta_df["obs_idx"]
    dataset_df["candidate_arc_id"] = meta_df["candidate_arc_id"]
    
    # Save as compressed parquet and csv
    parquet_path = os.path.join(output_dir, f"{partition}_dataset.parquet")
    dataset_df.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}")
    
    return {
        "partition": partition,
        "num_trips": len(trip_ids),
        "total_points": total_pts,
        "total_rows": len(full_X),
        "positive_count": int(full_y.sum()),
        "overall_cmr_pct": round(overall_cmr, 3),
        "mean_perp_dist_gt": float(np.mean([s["mean_perp_dist"] for s in alignment_stats])),
        "outlier_pct_gt": float(np.mean([s["outlier_pct"] for s in alignment_stats]))
    }


def main():
    print("=" * 60)
    print("STARTING PHASES 3 & 4: DATA PREPARATION & CANDIDATE GENERATION")
    print("=" * 60)
    
    base_data_dir = "data/Student folder"
    processed_dir = "data/processed"
    logs_dir = "results/logs"
    tables_dir = "results/tables"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    train_summary = process_partition("train", base_data_dir, processed_dir)
    test_summary = process_partition("student_test", base_data_dir, processed_dir)
    
    # Log summary
    prep_summary = [train_summary, test_summary]
    prep_df = pd.DataFrame(prep_summary)
    prep_csv = os.path.join(tables_dir, "data_preparation_summary.csv")
    prep_df.to_csv(prep_csv, index=False)
    print(f"\nData preparation summary saved to {prep_csv}")
    print(prep_df)


if __name__ == "__main__":
    main()
