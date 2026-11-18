"""
Emission Probability Modeling for PathSnap HMM.

Calculates observation emission probabilities by combining calibrated ML classifier
confidence with spatial perpendicular Gaussian error.

Mathematical formulation:
log P(z_t | s_i) = alpha * log(max(P_clf(s_i), eps)) - (d_perp(z_t, s_i)^2) / (2 * sigma_z^2)

- sigma_z: standard deviation of GPS measurement noise (typically 4.0 - 5.0m in urban canyons)
- alpha: relative weight assigned to classifier probability vs spatial perpendicular error
"""

import math
from typing import Dict, Any


class EmissionModel:
    """
    Computes log emission probabilities for road-segment candidate states.
    """
    def __init__(
        self,
        sigma_z: float = 4.5,            # GPS measurement error std dev (meters)
        classifier_weight: float = 2.8,  # Relative importance of classifier confidence
        min_clf_prob: float = 1e-5       # Prevents log(0) numerical underflow
    ):
        self.sigma_z = sigma_z
        self.classifier_weight = classifier_weight
        self.min_clf_prob = min_clf_prob
        self.two_sigma_sq = 2.0 * (sigma_z ** 2)

    def log_emission_probability(
        self,
        candidate: Dict[str, Any],
        clf_prob: float
    ) -> float:
        """
        Computes log P(z_t | s_i).
        
        Args:
            candidate: Dictionary containing candidate metadata (must include 'perp_dist')
            clf_prob: Calibrated classifier predicted probability in [0, 1]
            
        Returns:
            Scalar log emission probability.
        """
        perp_dist = candidate["perp_dist"]
        
        # Spatial perpendicular Gaussian log probability
        spatial_log_p = -(perp_dist ** 2) / self.two_sigma_sq
        
        # Classifier probability in log space
        clf_p_safe = max(float(clf_prob), self.min_clf_prob)
        clf_log_p = math.log(clf_p_safe)
        
        return float(self.classifier_weight * clf_log_p + spatial_log_p)
