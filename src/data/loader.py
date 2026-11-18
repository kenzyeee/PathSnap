"""
Trip Data Loading and Spatial Preparation Module for PathSnap.

Loads .track, .nodes, .arcs, and (strictly in isolated training/evaluation) .route files.
Computes local metric cartesian coordinates using Azimuthal Equidistant projection.
"""

import os
import math
import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple
from src.data.geo_utils import LocalProjector, calculate_bearing


class TripData:
    """
    Encapsulates all geographical and topological data for a single trip.
    """
    def __init__(
        self,
        trip_id: str,
        track_points: np.ndarray,      # (N, 3): lon, lat, time
        nodes: np.ndarray,             # (M, 2): lon, lat
        arcs: np.ndarray,              # (K, 2): from_node, to_node
        route: Optional[List[int]] = None, # (R,): ground-truth arc sequence
        trip_dir: Optional[str] = None
    ):
        self.trip_id = trip_id
        self.track = track_points      # N x 3 (lon, lat, ts)
        self.nodes = nodes            # M x 2 (lon, lat)
        self.arcs = arcs              # K x 2 (from_node, to_node)
        self.route = route            # ground truth arc_ids if present
        self.trip_dir = trip_dir
        
        # Center projection on mean of track
        center_lon = float(np.mean(self.track[:, 0]))
        center_lat = float(np.mean(self.track[:, 1]))
        self.projector = LocalProjector(center_lon, center_lat)
        
        # Project track points to meters (x, y)
        tx, ty = self.projector.to_meters_batch(self.track[:, 0], self.track[:, 1])
        self.projected_track = np.column_stack([tx, ty])
        
        # Project nodes to meters (x, y)
        nx_m, ny_m = self.projector.to_meters_batch(self.nodes[:, 0], self.nodes[:, 1])
        self.projected_nodes = np.column_stack([nx_m, ny_m])
        
        # Arc geometries in meters
        u_nodes = self.arcs[:, 0]
        v_nodes = self.arcs[:, 1]
        self.arc_starts = self.projected_nodes[u_nodes]  # K x 2
        self.arc_ends = self.projected_nodes[v_nodes]    # K x 2
        
        # Precompute arc vectors, lengths, and bearings
        arc_vecs = self.arc_ends - self.arc_starts
        self.arc_lengths = np.hypot(arc_vecs[:, 0], arc_vecs[:, 1])
        
        # Bearings in degrees [0, 360)
        # using geographical coords (lon, lat) for spherical bearing
        u_lon = self.nodes[u_nodes, 0]
        u_lat = self.nodes[u_nodes, 1]
        v_lon = self.nodes[v_nodes, 0]
        v_lat = self.nodes[v_nodes, 1]
        
        phi1 = np.radians(u_lat)
        phi2 = np.radians(v_lat)
        dlam = np.radians(v_lon - u_lon)
        y = np.sin(dlam) * np.cos(phi2)
        x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
        self.arc_bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        
        self._graph: Optional[nx.DiGraph] = None

    def get_graph(self) -> nx.DiGraph:
        """Constructs and caches the NetworkX directed graph of the road network."""
        if self._graph is None:
            G = nx.DiGraph()
            G.add_nodes_from(range(len(self.nodes)))
            for arc_id, (u, v) in enumerate(self.arcs):
                length = float(self.arc_lengths[arc_id])
                G.add_edge(int(u), int(v), arc_id=arc_id, weight=max(length, 1.0))
            self._graph = G
        return self._graph


def load_trip_data(
    trip_id: str,
    base_dir: str,
    load_route: bool = False
) -> TripData:
    """
    Loads trip data from a directory.
    NOTE: load_route defaults to False to strictly prevent data leakage in prediction paths.
    """
    trip_dir = os.path.join(base_dir, trip_id)
    if not os.path.exists(trip_dir):
        raise FileNotFoundError(f"Trip folder not found: {trip_dir}")
        
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    # 1. Load track (lon, lat, ts)
    track_list = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                track_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
    track = np.array(track_list, dtype=np.float64)
    
    # 2. Load nodes (lon, lat)
    nodes_list = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nodes_list.append([float(parts[0]), float(parts[1])])
    nodes = np.array(nodes_list, dtype=np.float64)
    
    # 3. Load arcs (u, v)
    arcs_list = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                arcs_list.append([int(parts[0]), int(parts[1])])
    arcs = np.array(arcs_list, dtype=np.int32)
    
    # 4. Load route if requested and available
    route = None
    if load_route and os.path.exists(route_path):
        route_list = []
        with open(route_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    route_list.append(int(stripped))
        route = route_list
        
    return TripData(
        trip_id=trip_id,
        track_points=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        trip_dir=trip_dir
    )
