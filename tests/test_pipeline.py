"""
Unit and Integration Test Suite for PathSnap.

Covers:
1. Spatial projections and distance calculations (no degree-Euclidean distance)
2. Feature extraction consistency and shape checks
3. HMM Viterbi decoding on a synthetic toy road network
4. Data leakage barrier: verifies prediction pipeline functions without .route
"""

import math
import numpy as np
import pytest
import networkx as nx

from src.data.geo_utils import (
    LocalProjector,
    haversine_distance,
    calculate_bearing,
    angle_diff,
    point_to_segment_projection_2d
)
from src.data.loader import TripData, load_trip_data
from src.features.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_trip, FEATURE_COLUMNS
from src.hmm.viterbi import HMMMapMatcher
from src.hmm.route_reconstructor import reconstruct_continuous_route


def test_local_projector_roundtrip():
    """Tests that LocalProjector forward and inverse projections preserve coordinates."""
    center_lon, center_lat = 8.58, 49.85
    proj = LocalProjector(center_lon, center_lat)
    
    # Test point
    test_lon, test_lat = 8.585, 49.855
    x, y = proj.to_meters(test_lon, test_lat)
    lon_back, lat_back = proj.to_lonlat(x, y)
    
    assert math.isclose(test_lon, lon_back, abs_tol=1e-6)
    assert math.isclose(test_lat, lat_back, abs_tol=1e-6)
    assert math.hypot(x, y) > 0.0


def test_point_to_segment_projection():
    """Tests 2D orthogonal segment projection."""
    # Horizontal segment from (0, 0) to (100, 0)
    ax, ay = 0.0, 0.0
    bx, by = 100.0, 0.0
    
    # Point at (50, 10): perp_dist should be 10, u should be 0.5
    d, u, px, py = point_to_segment_projection_2d(50.0, 10.0, ax, ay, bx, by)
    assert math.isclose(d, 10.0, abs_tol=1e-5)
    assert math.isclose(u, 0.5, abs_tol=1e-5)
    assert math.isclose(px, 50.0, abs_tol=1e-5)
    assert math.isclose(py, 0.0, abs_tol=1e-5)

    # Point beyond end at (120, 0): clamped to B, dist=20, u=1.0
    d, u, px, py = point_to_segment_projection_2d(120.0, 0.0, ax, ay, bx, by)
    assert math.isclose(d, 20.0, abs_tol=1e-5)
    assert math.isclose(u, 1.0, abs_tol=1e-5)
    assert math.isclose(px, 100.0, abs_tol=1e-5)


def test_viterbi_on_toy_graph():
    """
    Tests HMM Viterbi decoding on a toy 3-node graph:
    Arc 0: 0 -> 1 (connected to Arc 1)
    Arc 1: 1 -> 2
    Arc 2: 0 -> 2 (parallel direct shortcut)
    Arc 3: 2 -> 0 (loop back)
    """
    nodes = np.array([
        [8.000, 50.000],  # 0
        [8.001, 50.000],  # 1
        [8.002, 50.000]   # 2
    ], dtype=np.float64)
    
    arcs = np.array([
        [0, 1],  # Arc 0
        [1, 2],  # Arc 1
        [0, 2],  # Arc 2
        [2, 0]   # Arc 3
    ], dtype=np.int32)
    
    track = np.array([
        [8.0001, 50.0000, 1.0],
        [8.0005, 50.0000, 2.0],
        [8.0011, 50.0000, 3.0],
        [8.0015, 50.0000, 4.0]
    ], dtype=np.float64)

    trip = TripData("toy_trip", track, nodes, arcs, route=None)
    hmm = HMMMapMatcher(sigma_z=5.0, beta=3.0, classifier_weight=2.0)

    # Build candidates: for each observation, provide Arc 0 and Arc 1 with probabilities
    candidates = [
        [
            {"arc_id": 0, "perp_dist": 1.0, "u_clamped": 0.1, "arc_length": 50.0, "from_node": 0, "to_node": 1},
            {"arc_id": 2, "perp_dist": 5.0, "u_clamped": 0.05, "arc_length": 100.0, "from_node": 0, "to_node": 2}
        ],
        [
            {"arc_id": 0, "perp_dist": 1.0, "u_clamped": 0.5, "arc_length": 50.0, "from_node": 0, "to_node": 1},
            {"arc_id": 2, "perp_dist": 5.0, "u_clamped": 0.25, "arc_length": 100.0, "from_node": 0, "to_node": 2}
        ],
        [
            {"arc_id": 1, "perp_dist": 1.0, "u_clamped": 0.1, "arc_length": 50.0, "from_node": 1, "to_node": 2},
            {"arc_id": 2, "perp_dist": 5.0, "u_clamped": 0.55, "arc_length": 100.0, "from_node": 0, "to_node": 2}
        ],
        [
            {"arc_id": 1, "perp_dist": 1.0, "u_clamped": 0.5, "arc_length": 50.0, "from_node": 1, "to_node": 2},
            {"arc_id": 2, "perp_dist": 5.0, "u_clamped": 0.75, "arc_length": 100.0, "from_node": 0, "to_node": 2}
        ]
    ]
    probs = [
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1]
    ]

    decoded_arcs = hmm.decode(trip, candidates, probs)
    assert len(decoded_arcs) == 4
    # Expected sequence: Arc 0 for obs 0, 1; Arc 1 for obs 2, 3
    assert decoded_arcs == [0, 0, 1, 1]


def test_no_ground_truth_leakage():
    """
    Verifies that TripData and CandidateGenerator run successfully when
    route is None (as is strictly required in the live prediction pipeline).
    """
    trip = load_trip_data("00000000", "data/Student folder/student_test", load_route=False)
    assert trip.route is None
    
    cg = CandidateGenerator(trip, search_radius_m=65.0, max_candidates=10)
    candidates = cg.generate_all_candidates()
    assert len(candidates) == len(trip.track)
    assert len(candidates[0]) > 0
    
    X_df, y, meta = extract_features_for_trip(trip, candidates, reference_arcs=None)
    assert X_df.shape[1] == len(FEATURE_COLUMNS)
    assert not X_df.isnull().values.any()
    assert len(y) == len(X_df)


def test_route_reconstruction_and_continuity():
    """
    Tests route deduplication, gap bridging, and GeoJSON generation.
    """
    nodes = np.array([
        [8.0, 50.0],
        [8.0, 50.01],
        [8.0, 50.02],
        [8.0, 50.03]
    ], dtype=np.float64)
    arcs = np.array([
        [0, 1],  # Arc 0
        [1, 2],  # Arc 1
        [2, 3]   # Arc 2
    ], dtype=np.int32)
    track = np.array([[8.0, 50.0, 1.0]], dtype=np.float64)
    trip = TripData("test_trip", track, nodes, arcs)

    # Observations matched to [0, 0, 1, 1, 2]
    point_arcs = [0, 0, 1, 1, 2]
    result = reconstruct_continuous_route(trip, point_arcs)

    # Deduplicated route should be [0, 1, 2]
    assert result["route_arcs"] == [0, 1, 2]
    assert result["continuity_pct"] == 100.0
    assert result["gaps_found"] == 0
    assert len(result["coordinates"]) == 6  # 2 endpoints per arc
    assert result["geojson"]["type"] == "FeatureCollection"
    assert len(result["geojson"]["features"]) == 3


def test_lambdarank_model_ranking_and_calibration():
    """
    Tests LambdaRank model training, prediction, and post-training lambda scaling.
    """
    from src.models.lambdarank import LambdaRankModel, LambdaRankEnsembleModel

    # 4 queries of 5 candidates each
    X = np.random.RandomState(42).randn(20, len(FEATURE_COLUMNS))
    y = np.array([1, 0, 0, 0, 0,  0, 1, 0, 0, 0,  0, 0, 1, 0, 0,  1, 0, 0, 0, 0], dtype=np.int32)
    group = [5, 5, 5, 5]

    model = LambdaRankModel(n_estimators=10, random_state=42)
    model.fit(X, y, group=group)

    scores = model.predict(X)
    assert len(scores) == 20

    # Test calibrated probabilities with lambda parameter
    model.set_optimized_lambda(0.60)
    assert model.optimized_lambda_ == 0.60
    probs = model.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


