"""
Road-Network Aware Transition Probability Modeling for PathSnap HMM.

Implements the Newson-Krumm exponential transition model:
P(s_j | s_i) = (1 / beta) * exp( - | d_graph(s_i, s_j) - d_euclid(z_{t-1}, z_t) | / beta )

Key capabilities:
1. Same-arc displacement tracking: ensures forward movement along the arc.
2. Direct successor detection: topological connection v(s_i) == u(s_j).
3. Graph shortest-path distance: calculates true drivable network distance between arcs.
4. Large time gap handling: resets transition penalty when delta_t > threshold (e.g. signal loss).
5. Disconnected arc penalty: assigns small epsilon (log(1e-6) = -13.815) to impossible transitions.
"""

import math
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, Optional


class TransitionModel:
    """
    Computes log transition probabilities between candidate road segments.
    """
    def __init__(
        self,
        beta: float = 2.5,                 # Transition discrepancy parameter in meters
        max_graph_dist: float = 200.0,     # Max search distance for graph shortest paths
        large_gap_thresh_s: float = 12.0,  # Time delta threshold to reset transition chain
        unreachable_log_penalty: float = -50.0  # Heavy penalty for topologically disconnected transitions
    ):
        self.beta = beta
        self.max_graph_dist = max_graph_dist
        self.large_gap_thresh_s = large_gap_thresh_s
        self.unreachable_log_penalty = unreachable_log_penalty
        self._sp_cache: Dict[Tuple[int, int], float] = {}

    def clear_cache(self):
        """Clears the shortest-path distance cache between trips."""
        self._sp_cache.clear()

    def log_transition_probability(
        self,
        c_prev: Dict[str, Any],
        c_curr: Dict[str, Any],
        gps_dist: float,
        dt: float,
        graph: nx.DiGraph
    ) -> float:
        """
        Computes log P(s_j | s_i).
        
        Args:
            c_prev: Candidate dictionary at step t-1
            c_curr: Candidate dictionary at step t
            gps_dist: Great-circle / Euclidean displacement between GPS points (meters)
            dt: Time difference between consecutive GPS observations (seconds)
            graph: Road network directed graph (NetworkX DiGraph)
            
        Returns:
            Scalar log transition probability.
        """
        # If time gap is unusually large (> 12s), the vehicle could be anywhere;
        # reset transition chain with a mild uniform prior to avoid false penalization.
        if dt > self.large_gap_thresh_s:
            return -2.0

        arc_prev = c_prev["arc_id"]
        arc_curr = c_curr["arc_id"]

        # Case 1: Vehicle remained on the same arc
        if arc_prev == arc_curr:
            delta_along = (
                c_curr["u_clamped"] * c_curr["arc_length"] -
                c_prev["u_clamped"] * c_prev["arc_length"]
            )
            if delta_along >= -5.0:
                net_dist = max(0.0, delta_along)
            else:
                # Reverse movement along a directed arc (penalize backward travel)
                net_dist = abs(delta_along) + 30.0
                
            diff = abs(net_dist - gps_dist)
            return float(-diff / self.beta)

        # Case 2: Candidate arcs are distinct
        v_prev = c_prev["to_node"]
        u_curr = c_curr["from_node"]

        remaining_prev = (1.0 - c_prev["u_clamped"]) * c_prev["arc_length"]
        initial_curr = c_curr["u_clamped"] * c_curr["arc_length"]

        # 2a. Direct successor (v_prev == u_curr)
        if v_prev == u_curr:
            net_dist = remaining_prev + initial_curr
            diff = abs(net_dist - gps_dist)
            return float(-diff / self.beta)

        # 2b. Multi-hop shortest path in directed graph with distance cutoff
        cache_key = (v_prev, u_curr)
        if cache_key in self._sp_cache:
            path_len = self._sp_cache[cache_key]
        else:
            try:
                length, _ = nx.single_source_dijkstra(
                    graph,
                    source=v_prev,
                    target=u_curr,
                    cutoff=self.max_graph_dist,
                    weight="weight"
                )
                path_len = length
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                path_len = np.inf
            self._sp_cache[cache_key] = path_len

        if np.isfinite(path_len) and path_len <= self.max_graph_dist:
            net_dist = remaining_prev + path_len + initial_curr
            diff = abs(net_dist - gps_dist)
            return float(-diff / self.beta)

        # Case 3: Disconnected or out of reasonable reach
        return self.unreachable_log_penalty
