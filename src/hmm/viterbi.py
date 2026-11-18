"""
Viterbi Sequence Decoding Module for PathSnap.

Orchestrates the EmissionModel and TransitionModel using the dynamic programming
Viterbi algorithm in log-space with backpointer tracking.

Produces the globally optimal sequence of road segments that maximizes joint likelihood.
"""

import math
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

from src.data.loader import TripData
from src.hmm.emission import EmissionModel
from src.hmm.transition import TransitionModel


class HMMMapMatcher:
    """
    Decodes the most likely sequence of road segments using the Viterbi algorithm.
    """
    def __init__(
        self,
        sigma_z: float = 5.0,
        beta: float = 2.5,
        classifier_weight: float = 3.5,
        max_graph_dist: float = 200.0,
        large_gap_thresh_s: float = 12.0
    ):
        self.emission_model = EmissionModel(
            sigma_z=sigma_z,
            classifier_weight=classifier_weight
        )
        self.transition_model = TransitionModel(
            beta=beta,
            max_graph_dist=max_graph_dist,
            large_gap_thresh_s=large_gap_thresh_s
        )

    def decode(
        self,
        trip: TripData,
        candidates_per_obs: List[List[Dict[str, Any]]],
        candidate_probs_per_obs: List[List[float]]
    ) -> List[int]:
        """
        Runs the Viterbi algorithm and returns the sequence of optimal arc_ids (length N).
        """
        N = len(trip.track)
        if N == 0:
            return []

        graph = trip.get_graph()
        self.transition_model.clear_cache()

        # V[t][i]: max log probability of path ending at candidate i at step t
        # backpointer[t][i]: index of predecessor candidate at step t-1
        V: List[np.ndarray] = []
        backpointers: List[np.ndarray] = []

        # Step 0: Initialize with emission probabilities
        cands_0 = candidates_per_obs[0]
        probs_0 = candidate_probs_per_obs[0]
        v0 = np.array([
            self.emission_model.log_emission_probability(c, p)
            for c, p in zip(cands_0, probs_0)
        ], dtype=np.float32)
        V.append(v0)

        # Forward Recurrence
        for t in range(1, N):
            cands_prev = candidates_per_obs[t - 1]
            cands_curr = candidates_per_obs[t]
            probs_curr = candidate_probs_per_obs[t]

            # GPS displacement and time gap
            p_prev = trip.projected_track[t - 1]
            p_curr = trip.projected_track[t]
            gps_dist = float(math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]))
            dt = float(trip.track[t, 2] - trip.track[t - 1, 2])

            v_curr = np.zeros(len(cands_curr), dtype=np.float32)
            bp_curr = np.zeros(len(cands_curr), dtype=np.int32)

            for j, c_j in enumerate(cands_curr):
                emission_log_p = self.emission_model.log_emission_probability(c_j, probs_curr[j])
                best_log_p = -np.inf
                best_prev_idx = 0

                for i, c_i in enumerate(cands_prev):
                    trans_log_p = self.transition_model.log_transition_probability(
                        c_i, c_j, gps_dist, dt, graph
                    )
                    total_p = V[t - 1][i] + trans_log_p
                    if total_p > best_log_p:
                        best_log_p = total_p
                        best_prev_idx = i

                v_curr[j] = best_log_p + emission_log_p
                bp_curr[j] = best_prev_idx

            V.append(v_curr)
            backpointers.append(bp_curr)

        # Backtracking from the most likely final state
        best_final_idx = int(np.argmax(V[-1]))
        decoded_indices = [best_final_idx]

        for t in range(N - 1, 0, -1):
            prev_idx = backpointers[t - 1][decoded_indices[-1]]
            decoded_indices.append(prev_idx)

        decoded_indices.reverse()
        decoded_arcs = [
            candidates_per_obs[t][idx]["arc_id"]
            for t, idx in enumerate(decoded_indices)
        ]
        return decoded_arcs
