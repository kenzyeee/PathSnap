"""
Dataset Inventory and Exploration Module for PathSnap.

Audits raw trips in both train and student_test partitions:
- File availability and byte sizes
- Observation counts and sampling rate statistics (dt)
- Geographic bounding boxes and plausibility
- Road network node/arc metrics, degree distribution, and connectivity
- Generates summary CSVs for report documentation
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def analyze_track_file(track_path: str) -> Dict[str, Any]:
    """Inspects a .track file: row count, time gaps, speeds, bounding box."""
    data = []
    with open(track_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))
    
    if not data:
        return {
            "num_points": 0,
            "min_lon": np.nan, "max_lon": np.nan,
            "min_lat": np.nan, "max_lat": np.nan,
            "duration_s": 0.0,
            "dt_1s_pct": 0.0,
            "max_gap_s": 0.0,
            "gaps_gt_5s": 0,
        }
    
    arr = np.array(data)
    lons = arr[:, 0]
    lats = arr[:, 1]
    times = arr[:, 2]
    
    dt = np.diff(times)
    num_dt = len(dt)
    dt_1s = np.sum(np.isclose(dt, 1.0, atol=0.05)) if num_dt > 0 else 0
    dt_1s_pct = (dt_1s / num_dt * 100.0) if num_dt > 0 else 0.0
    max_gap = float(np.max(dt)) if num_dt > 0 else 0.0
    gaps_gt_5s = int(np.sum(dt > 5.0)) if num_dt > 0 else 0
    
    return {
        "num_points": len(arr),
        "min_lon": float(np.min(lons)),
        "max_lon": float(np.max(lons)),
        "min_lat": float(np.min(lats)),
        "max_lat": float(np.max(lats)),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "dt_1s_pct": round(dt_1s_pct, 2),
        "max_gap_s": round(max_gap, 2),
        "gaps_gt_5s": gaps_gt_5s,
    }


def analyze_network_files(nodes_path: str, arcs_path: str, route_path: str = None) -> Dict[str, Any]:
    """Inspects road network nodes and arcs."""
    # Nodes count
    node_count = 0
    with open(nodes_path, "r", encoding="utf-8") as f:
        for _ in f:
            node_count += 1
            
    # Arcs and degrees
    arc_count = 0
    in_degrees: Dict[int, int] = {}
    out_degrees: Dict[int, int] = {}
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                out_degrees[u] = out_degrees.get(u, 0) + 1
                in_degrees[v] = in_degrees.get(v, 0) + 1
                arc_count += 1
                
    # Route count if available
    route_len = 0
    if route_path and os.path.exists(route_path):
        with open(route_path, "r", encoding="utf-8") as f:
            for _ in f:
                route_len += 1
                
    active_nodes = set(in_degrees.keys()).union(set(out_degrees.keys()))
    isolated_nodes = max(0, node_count - len(active_nodes))
    
    return {
        "num_nodes": node_count,
        "num_arcs": arc_count,
        "num_route_arcs": route_len,
        "isolated_nodes": isolated_nodes,
        "avg_out_degree": round(arc_count / node_count, 3) if node_count > 0 else 0.0,
    }


def generate_inventory(data_dir: str, output_csv: str) -> pd.DataFrame:
    """Scans all trips in data_dir (both train and student_test) and saves an inventory CSV."""
    records = []
    
    for partition in ["train", "student_test"]:
        part_dir = os.path.join(data_dir, partition)
        if not os.path.exists(part_dir):
            continue
        
        trip_ids = sorted([
            d for d in os.listdir(part_dir)
            if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")
        ])
        
        for tid in trip_ids:
            trip_path = os.path.join(part_dir, tid)
            track_file = os.path.join(trip_path, f"{tid}.track")
            nodes_file = os.path.join(trip_path, f"{tid}.nodes")
            arcs_file = os.path.join(trip_path, f"{tid}.arcs")
            route_file = os.path.join(trip_path, f"{tid}.route")
            
            # File sizes in KB
            track_sz = os.path.getsize(track_file) / 1024.0 if os.path.exists(track_file) else 0.0
            nodes_sz = os.path.getsize(nodes_file) / 1024.0 if os.path.exists(nodes_file) else 0.0
            arcs_sz = os.path.getsize(arcs_file) / 1024.0 if os.path.exists(arcs_file) else 0.0
            route_sz = os.path.getsize(route_file) / 1024.0 if os.path.exists(route_file) else 0.0
            
            track_stats = analyze_track_file(track_file) if os.path.exists(track_file) else {}
            net_stats = analyze_network_files(nodes_file, arcs_file, route_file) if os.path.exists(nodes_file) else {}
            
            rec = {
                "partition": partition,
                "trip_id": tid,
                "track_size_kb": round(track_sz, 2),
                "nodes_size_kb": round(nodes_sz, 2),
                "arcs_size_kb": round(arcs_sz, 2),
                "route_size_kb": round(route_sz, 2),
                "total_size_kb": round(track_sz + nodes_sz + arcs_sz + route_sz, 2),
                **track_stats,
                **net_stats,
            }
            records.append(rec)
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
