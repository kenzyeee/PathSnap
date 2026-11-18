"""
Feature Extraction Module for PathSnap Candidate Arc Classification.

Computes discriminatory geometric, kinematic, and topological features
for every (observation, candidate) pair.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from src.data.loader import TripData
from src.data.geo_utils import calculate_bearing, angle_diff


FEATURE_COLUMNS = [
    "perp_dist",
    "proj_fraction",
    "proj_dist_along",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "gps_speed",
    "gps_accel",
    "bearing_diff",
    "bearing_cosine",
    "dist_rank",
    "rel_dist",
    "num_candidates",
    "time_gap",
    "turn_angle",
    "connected_to_prev_closest",
    "start_in_degree",
    "end_out_degree"
]


def compute_kinematics(track_xy: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes speed (m/s), acceleration (m/s^2), and GPS headings (deg) for a trajectory.
    """
    N = len(track_xy)
    speeds = np.zeros(N, dtype=np.float32)
    accels = np.zeros(N, dtype=np.float32)
    headings = np.zeros(N, dtype=np.float32)
    turn_angles = np.zeros(N, dtype=np.float32)
    
    if N < 2:
        return {"speeds": speeds, "accels": accels, "headings": headings, "turn_angles": turn_angles}
        
    dpos = np.diff(track_xy, axis=0)  # (N-1, 2)
    dists = np.hypot(dpos[:, 0], dpos[:, 1])  # (N-1,)
    dt = np.diff(timestamps)  # (N-1,)
    dt_safe = np.maximum(dt, 0.01)
    
    step_speeds = dists / dt_safe
    speeds[1:] = step_speeds
    speeds[0] = speeds[1]
    
    # Acceleration
    if N > 2:
        dv = np.diff(speeds)
        step_accels = dv / dt_safe
        accels[1:] = step_accels
        accels[0] = accels[1]
        
    # Headings from displacement vectors: dx, dy
    # dy is North, dx is East in local AEQD projection
    step_headings = (np.degrees(np.arctan2(dpos[:, 0], dpos[:, 1])) + 360.0) % 360.0
    headings[1:] = step_headings
    headings[0] = headings[1]
    
    # Turn angles: |heading[t] - heading[t-1]|
    for t in range(1, N):
        diff = abs(headings[t] - headings[t - 1]) % 360.0
        turn_angles[t] = 360.0 - diff if diff > 180.0 else diff
        
    return {
        "speeds": speeds,
        "accels": accels,
        "headings": headings,
        "turn_angles": turn_angles
    }


def extract_features_for_trip(
    trip: TripData,
    candidates_per_obs: List[List[Dict[str, Any]]],
    reference_arcs: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts features for all (observation, candidate) pairs in a trip.
    
    Returns:
        - X_df: DataFrame with FEATURE_COLUMNS
        - y: binary numpy array (1 if candidate == reference_arc, 0 otherwise)
        - metadata: list of dicts with (trip_id, obs_idx, candidate_arc_id)
    """
    N = len(trip.track)
    kinematics = compute_kinematics(trip.projected_track, trip.track[:, 2])
    
    # Compute node degrees for topology features
    in_degrees = {}
    out_degrees = {}
    for arc_id, (u, v) in enumerate(trip.arcs):
        out_degrees[u] = out_degrees.get(u, 0) + 1
        in_degrees[v] = in_degrees.get(v, 0) + 1
        
    rows = []
    labels = []
    meta = []
    
    prev_closest_arc_id = None
    prev_closest_to_node = None
    
    for t in range(N):
        px, py = trip.projected_track[t]
        ts = trip.track[t, 2]
        time_gap = float(trip.track[t, 2] - trip.track[t - 1, 2]) if t > 0 else 1.0
        
        cands = candidates_per_obs[t]
        if not cands:
            continue
            
        min_perp_dist = cands[0]["perp_dist"]
        num_cands = len(cands)
        gps_spd = float(kinematics["speeds"][t])
        gps_acc = float(kinematics["accels"][t])
        gps_hdg = float(kinematics["headings"][t])
        turn_ang = float(kinematics["turn_angles"][t])
        
        true_arc = reference_arcs[t] if reference_arcs is not None else None
        
        for rank, c in enumerate(cands):
            arc_id = c["arc_id"]
            perp_dist = c["perp_dist"]
            u_clamped = c["u_clamped"]
            arc_len = c["arc_length"]
            arc_bearing = c["arc_bearing"]
            from_node = c["from_node"]
            to_node = c["to_node"]
            
            # Distance from point to start and end nodes of arc
            ax, ay = trip.arc_starts[arc_id]
            bx, by = trip.arc_ends[arc_id]
            dist_to_start = math.hypot(px - ax, py - ay)
            dist_to_end = math.hypot(px - bx, py - by)
            
            # Bearing difference and cosine alignment
            # When speed is very low (< 0.5 m/s), heading is noisy, so alignment is less reliable
            b_diff = angle_diff(gps_hdg, arc_bearing) if gps_spd > 0.5 else 90.0
            b_cos = math.cos(math.radians(b_diff)) if gps_spd > 0.5 else 0.0
            
            # Connectivity to previous observation's top candidate
            connected_to_prev = 0
            if prev_closest_arc_id is not None:
                if arc_id == prev_closest_arc_id or from_node == prev_closest_to_node:
                    connected_to_prev = 1
                    
            feat = {
                "perp_dist": perp_dist,
                "proj_fraction": u_clamped,
                "proj_dist_along": u_clamped * arc_len,
                "dist_to_start": dist_to_start,
                "dist_to_end": dist_to_end,
                "arc_length": arc_len,
                "gps_speed": gps_spd,
                "gps_accel": gps_acc,
                "bearing_diff": b_diff,
                "bearing_cosine": b_cos,
                "dist_rank": float(rank),
                "rel_dist": perp_dist - min_perp_dist,
                "num_candidates": float(num_cands),
                "time_gap": time_gap,
                "turn_angle": turn_ang,
                "connected_to_prev_closest": float(connected_to_prev),
                "start_in_degree": float(in_degrees.get(from_node, 0)),
                "end_out_degree": float(out_degrees.get(to_node, 0))
            }
            rows.append(feat)
            
            if true_arc is not None:
                labels.append(1 if arc_id == true_arc else 0)
            else:
                labels.append(0)
                
            meta.append({
                "trip_id": trip.trip_id,
                "obs_idx": t,
                "candidate_arc_id": arc_id,
                "perp_dist": perp_dist
            })
            
        # Update prev closest candidate for next step
        prev_closest_arc_id = cands[0]["arc_id"]
        prev_closest_to_node = cands[0]["to_node"]
        
    X_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = np.array(labels, dtype=np.int32)
    return X_df, y, meta
