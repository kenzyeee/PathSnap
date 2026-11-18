"""
Ground-Truth Monotonic Alignment Module for PathSnap.

Derives per-observation reference arc labels by aligning GPS trajectory points
monotonically to the ground-truth .route sequence using constrained Dynamic Programming.

CRITICAL SEPARATION RULE:
This module is ONLY used during offline training label preparation and evaluation.
The live prediction pipeline (candidate generation, features, model, HMM) must NEVER call this module.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.data.loader import TripData
from src.data.geo_utils import point_to_segment_projection_2d


def align_gps_to_route_dp(
    trip: TripData,
    max_jump: int = 5,
    trans_weight: float = 2.0,
    outlier_dist_thresh: float = 35.0
) -> Dict[str, Any]:
    """
    Aligns GPS observations monotonically to the ground-truth route sequence R.
    Returns:
        - reference_arcs: list of arc_ids for each GPS observation (length N)
        - route_indices: route index k for each GPS observation (length N)
        - perp_distances: distance in meters to the assigned reference arc
        - outlier_mask: boolean mask indicating GPS points far from the assigned arc
    """
    if not trip.route:
        raise ValueError(f"Trip {trip.trip_id} has no ground-truth route to align.")

    route_arcs = np.array(trip.route, dtype=np.int32)
    K = len(route_arcs)
    N = len(trip.track)

    # Pre-extract arc coordinates in meters
    u_nodes = trip.arcs[route_arcs, 0]
    v_nodes = trip.arcs[route_arcs, 1]
    starts = trip.projected_nodes[u_nodes]  # K x 2
    ends = trip.projected_nodes[v_nodes]    # K x 2
    gps_xy = trip.projected_track           # N x 2

    # Precompute pairwise distance matrix between each GPS point and each route arc: N x K
    # Using vectorized point-to-segment distance
    dist_matrix = np.zeros((N, K), dtype=np.float32)
    
    seg_vecs = ends - starts  # K x 2
    seg_lens_sq = np.sum(seg_vecs ** 2, axis=1)  # K
    seg_lens_sq = np.maximum(seg_lens_sq, 1e-6)

    for t in range(N):
        px, py = gps_xy[t, 0], gps_xy[t, 1]
        # Vectors from start of each arc to P
        wx = px - starts[:, 0]
        wy = py - starts[:, 1]
        
        # Projection fraction u
        u = (wx * seg_vecs[:, 0] + wy * seg_vecs[:, 1]) / seg_lens_sq
        u_clamped = np.clip(u, 0.0, 1.0)
        
        # Projected point coordinates
        proj_x = starts[:, 0] + u_clamped * seg_vecs[:, 0]
        proj_y = starts[:, 1] + u_clamped * seg_vecs[:, 1]
        
        # Perpendicular distance
        dists = np.hypot(px - proj_x, py - proj_y)
        dist_matrix[t, :] = dists

    # Dynamic Programming:
    # DP[t, k]: minimum cumulative cost to match point t to route arc k
    # Monotonicity: next point must have route index >= current route index
    dp = np.full((N, K), np.inf, dtype=np.float32)
    backpointer = np.zeros((N, K), dtype=np.int32)

    # Initialize t = 0 (allow starting anywhere in the first few arcs)
    init_window = min(K, 10)
    for k in range(init_window):
        dp[0, k] = dist_matrix[0, k]

    # Recurrence
    for t in range(1, N):
        for k in range(K):
            # Candidate predecessors j: j <= k and k - j <= max_jump
            min_j = max(0, k - max_jump)
            best_val = np.inf
            best_j = min_j
            
            for j in range(min_j, k + 1):
                # Penalty for advancing arcs: 0 if j == k, trans_weight * (k - j)
                jump_penalty = 0.0 if j == k else trans_weight * float(k - j)
                val = dp[t - 1, j] + jump_penalty
                if val < best_val:
                    best_val = val
                    best_j = j
                    
            dp[t, k] = best_val + dist_matrix[t, k]
            backpointer[t, k] = best_j

    # Backtracking from the best final state in the last few arcs
    final_window_start = max(0, K - 15)
    best_final_k = final_window_start + int(np.argmin(dp[N - 1, final_window_start:]))

    aligned_route_indices = np.zeros(N, dtype=np.int32)
    aligned_route_indices[N - 1] = best_final_k
    
    curr_k = best_final_k
    for t in range(N - 1, 0, -1):
        curr_k = backpointer[t, curr_k]
        aligned_route_indices[t - 1] = curr_k

    aligned_arc_ids = route_arcs[aligned_route_indices]
    aligned_dists = dist_matrix[np.arange(N), aligned_route_indices]
    outlier_mask = aligned_dists > outlier_dist_thresh

    return {
        "reference_arcs": aligned_arc_ids.tolist(),
        "route_indices": aligned_route_indices.tolist(),
        "perp_distances": aligned_dists.tolist(),
        "outlier_mask": outlier_mask.tolist(),
        "mean_perp_dist": float(np.mean(aligned_dists)),
        "median_perp_dist": float(np.median(aligned_dists)),
        "max_perp_dist": float(np.max(aligned_dists)),
        "outlier_pct": float(np.mean(outlier_mask) * 100.0)
    }
