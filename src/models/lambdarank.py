"""
LambdaRank / LambdaMART Ranking Model for PathSnap.

Learning-to-Rank (LTR) model leveraging LightGBM LGBMRanker to optimize
Normalized Discounted Cumulative Gain (NDCG@1) across candidate road segments.

Instead of treating candidates as independent binary classification samples,
LambdaRank optimizes the ordering within each GPS observation's candidate set,
directly training the model to place the true road arc at Rank 1.

Post-training calibration optimizes the scaling factor lambda (temperature)
to calibrate ranking scores for log-likelihood emission modeling in the HMM.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from lightgbm import LGBMRanker
from sklearn.base import BaseEstimator, ClassifierMixin


class LambdaRankModel(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible LambdaRank / LambdaMART estimator for candidate ranking.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        lambdarank_truncation_level: int = 5,
        lambdarank_norm: bool = True,
        lambda_param: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambdarank_truncation_level = lambdarank_truncation_level
        self.lambdarank_norm = lambdarank_norm
        self.lambda_param = lambda_param
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.ranker_: Optional[LGBMRanker] = None
        self.classes_ = np.array([0, 1])
        self.feature_importances_: Optional[np.ndarray] = None
        self.optimized_lambda_: float = lambda_param

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        group: Union[List[int], np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        eval_group: Optional[List[List[int]]] = None,
        **kwargs
    ) -> "LambdaRankModel":
        """
        Fits the LambdaRank model on grouped candidate sets.
        
        Args:
            X: Feature matrix of shape (N_samples, N_features)
            y: Binary relevance labels (1 for true arc, 0 for distractors)
            group: List of query group lengths (candidates per observation)
            eval_set: Optional validation tuples (X_val, y_val)
            eval_group: Optional validation group lengths
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            
        y_vec = np.asarray(y, dtype=np.int32)
        group_vec = np.asarray(group, dtype=np.int32)

        self.ranker_ = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            lambdarank_truncation_level=self.lambdarank_truncation_level,
            lambdarank_norm=self.lambdarank_norm,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            importance_type="gain",
            verbose=-1
        )

        fit_params = {"eval_at": [1, 3]}
        if eval_set is not None and eval_group is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_group"] = eval_group

        self.ranker_.fit(X_mat, y_vec, group=group_vec, **fit_params)
        self.feature_importances_ = self.ranker_.feature_importances_
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes raw ranking scores for candidate segments."""
        if self.ranker_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.ranker_.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Converts raw ranking scores into calibrated probabilities using
        the optimized scaling parameter lambda (temperature scaling).
        
        P(c_i | s_i) = 1 / (1 + exp(- lambda * s_i))
        """
        scores = self.predict(X)
        scale = float(self.optimized_lambda_ if self.optimized_lambda_ is not None else self.lambda_param)
        
        # Numerically stable sigmoid
        scaled_scores = np.clip(scale * scores, -50.0, 50.0)
        prob_1 = 1.0 / (1.0 + np.exp(-scaled_scores))
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def set_optimized_lambda(self, optimal_lambda: float):
        """Sets the post-training optimized lambda parameter."""
        self.optimized_lambda_ = float(optimal_lambda)
        self.lambda_param = float(optimal_lambda)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambdarank_truncation_level": self.lambdarank_truncation_level,
            "lambdarank_norm": self.lambdarank_norm,
            "lambda_param": self.lambda_param,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self


class LambdaRankEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Hybrid Ensemble Ranker blending Calibrated HistGradientBoosting with Listwise LambdaRank.
    Combines the sharp candidate discrimination of LambdaRank (NDCG@1 optimization)
    with the smooth probability calibration of gradient boosted trees.
    
    P_ensemble = (1 - blend_weight) * P_HGB + blend_weight * P_LambdaRank
    """
    def __init__(
        self,
        lambdarank_model: Optional[LambdaRankModel] = None,
        hgb_model: Optional[Any] = None,
        blend_weight: float = 0.10,
        optimized_lambda: float = 0.60
    ):
        self.lambdarank_model = lambdarank_model
        self.hgb_model = hgb_model
        self.blend_weight = blend_weight
        self.optimized_lambda_ = optimized_lambda
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        p_lgb = self.lambdarank_model.predict_proba(X)[:, 1]
        p_hgb = self.hgb_model.predict_proba(X)[:, 1]
        p_blend = (1.0 - self.blend_weight) * p_hgb + self.blend_weight * p_lgb
        return np.column_stack([1.0 - p_blend, p_blend])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.5, 1, 0)

