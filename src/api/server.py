"""
FastAPI Backend Server for PathSnap (Runs on Laptop 1).

Hosts the trained classifier pipeline, spatial index, HMM Viterbi route decoder,
and server-side isolated evaluation process.

CRITICAL SECURITY & INTEGRITY RULE:
The prediction pipeline (load track/nodes/arcs -> candidates -> features -> model -> HMM)
NEVER reads the .route file. Ground-truth evaluation occurs strictly post-prediction
via the isolated src.eval module.
"""

import os
import time
import logging
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.data.loader import load_trip_data
from src.features.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_trip, FEATURE_COLUMNS
from src.hmm.viterbi import HMMMapMatcher
from src.hmm.route_reconstructor import reconstruct_continuous_route
from src.eval.evaluator import evaluate_predictions_against_ground_truth


# Configure Logging
LOG_FILE = "results/logs/api_requests.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("pathsnap_api")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# FastAPI App
app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Pes University Hackathon API for Distributed Map-Matching (Laptop 1)",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for artifacts
MODEL_PIPELINE = None
HMM_MATCHER = None
MODEL_INFO = {
    "model_name": "LambdaRank",
    "algorithm": "Learning-to-Rank (LambdaMART)",
    "optimized_lambda": 0.60,
    "ndcg_objective": "NDCG@1"
}
BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", "data/Student folder")
EVAL_DATA_DIR = os.getenv("EVAL_DATA_DIR", "data/Student folder/evaluator_test" if os.path.exists("data/Student folder/evaluator_test") else "data/Student folder/student_test")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/best_tuned_model.joblib")


def resolve_trip_dir(trip_id: str) -> tuple[str, str]:
    """Finds directory and partition for a given trip_id."""
    trip_id = trip_id.strip()
    # Check evaluator_test first, then student_test, then train
    for part in ["evaluator_test", "student_test", "train"]:
        p = os.path.join(BASE_DATA_DIR, part, trip_id)
        if os.path.exists(p):
            return p, part
    # Check fallback EVAL_DATA_DIR
    p = os.path.join(EVAL_DATA_DIR, trip_id)
    if os.path.exists(p):
        return p, os.path.basename(EVAL_DATA_DIR)
    raise FileNotFoundError(f"Trip ID '{trip_id}' not found in any data directory.")


@app.on_event("startup")
def load_artifacts():
    """Loads trained model pipeline and initializes HMM matcher at server startup."""
    global MODEL_PIPELINE, HMM_MATCHER, MODEL_INFO
    logger.info("Initializing PathSnap API Server...")
    
    candidate_paths = [
        MODEL_PATH,
        "artifacts/models/best_tuned_model.joblib",
        "artifacts/models/lambdarank_model.joblib",
        "artifacts/models/HistGradientBoosting.joblib",
        "artifacts/models/RandomForest.joblib",
        "artifacts/models/LogisticRegression.joblib"
    ]
    
    loaded = False
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                MODEL_PIPELINE = joblib.load(path)
                logger.info(f"Loaded trained model artifact from: {path}")
                loaded = True
                
                # Extract metadata
                cls_name = type(MODEL_PIPELINE).__name__
                if "LambdaRank" in cls_name:
                    MODEL_INFO["model_name"] = "LambdaRank (Optimized LambdaMART)"
                    MODEL_INFO["algorithm"] = "Learning-to-Rank (LTR)"
                    MODEL_INFO["optimized_lambda"] = getattr(MODEL_PIPELINE, "optimized_lambda_", 0.60)
                else:
                    MODEL_INFO["model_name"] = cls_name
                    MODEL_INFO["algorithm"] = "Calibrated Pointwise Classifier"
                    MODEL_INFO["optimized_lambda"] = 0.60
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                
    if not loaded:
        logger.warning("No pre-trained model artifact found at startup. Please train or tune models first.")
        
    # Initialize HMM matcher with calibrated parameters
    HMM_MATCHER = HMMMapMatcher(
        sigma_z=5.0,
        beta=2.5,
        classifier_weight=3.5,
        max_graph_dist=200.0,
        large_gap_thresh_s=12.0
    )
    HMM_MATCHER.transition_model.unreachable_log_penalty = -50.0
    logger.info(f"HMM Viterbi Decoder initialized. Default evaluation directory: {EVAL_DATA_DIR}")


# Request & Response Schemas
class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="ID of the trip to predict (e.g. '00000015' or '00000000')")
    include_ground_truth: bool = Field(False, description="Whether to include GT route coordinates in response (default: False)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    algorithm: Optional[str] = None
    optimized_lambda: Optional[float] = None
    eval_data_dir: str
    available_trips_count: int
    partitions: Dict[str, List[str]] = {}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint: verifies model readiness and available trips."""
    partitions = {}
    total_trips = 0
    for part in ["evaluator_test", "student_test"]:
        part_dir = os.path.join(BASE_DATA_DIR, part)
        if os.path.exists(part_dir):
            tids = sorted([
                d for d in os.listdir(part_dir)
                if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")
            ])
            partitions[part] = tids
            total_trips += len(tids)

    return HealthResponse(
        status="healthy",
        model_loaded=(MODEL_PIPELINE is not None),
        model_name=MODEL_INFO.get("model_name"),
        algorithm=MODEL_INFO.get("algorithm"),
        optimized_lambda=MODEL_INFO.get("optimized_lambda"),
        eval_data_dir=EVAL_DATA_DIR,
        available_trips_count=total_trips,
        partitions=partitions
    )


@app.get("/trips")
def list_trips(partition: Optional[str] = Query(None)):
    """Returns available trips across evaluator_test and student_test."""
    partitions = {}
    all_trips = []
    
    parts_to_scan = [partition] if partition else ["evaluator_test", "student_test"]
    for part in parts_to_scan:
        part_dir = os.path.join(BASE_DATA_DIR, part)
        if os.path.exists(part_dir):
            tids = sorted([
                d for d in os.listdir(part_dir)
                if os.path.isdir(os.path.join(part_dir, d)) and not d.startswith(".")
            ])
            partitions[part] = tids
            all_trips.extend(tids)
            
    return {
        "partitions": partitions,
        "trip_count": len(all_trips),
        "trips": sorted(list(set(all_trips)))
    }


@app.post("/predict")
def predict_trip(req: PredictRequest):
    """
    Executes full map-matching and route-reconstruction pipeline for a given trip.
    
    1. Loads .track, .nodes, .arcs (NEVER reads .route).
    2. Generates candidate arcs using multi-point spatial KDTree.
    3. Extracts 18 geometric, kinematic, and topological features.
    4. Predicts candidate probabilities using trained ML model.
    5. Performs global sequence optimization via log-space Viterbi HMM.
    6. Reconstructs deduplicated, continuous route geometry.
    7. Evaluates predictions against server-side .route (if present).
    """
    t_start = time.time()
    trip_id = req.trip_id.strip()
    try:
        trip_dir, partition_name = resolve_trip_dir(trip_id)
        parent_dir = os.path.dirname(trip_dir)
    except FileNotFoundError:
        logger.error(f"Trip not found: {trip_id} across {BASE_DATA_DIR}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip ID '{trip_id}' not found in {BASE_DATA_DIR}"
        )

    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on server."
        )

    # 1. Load Trip Data (load_route=False STRICTLY ENFORCED)
    try:
        trip = load_trip_data(trip_id, parent_dir, load_route=False)
    except Exception as e:
        logger.error(f"Failed to load trip {trip_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load trip files: {str(e)}"
        )

    # 2. Candidate Generation
    cg = CandidateGenerator(trip, search_radius_m=65.0, sample_spacing_m=20.0, max_candidates=15, min_candidates=3)
    candidates = cg.generate_all_candidates()

    # 3. Feature Extraction
    X_df, _, meta = extract_features_for_trip(trip, candidates, reference_arcs=None)

    # 4. Model Inference
    probs = MODEL_PIPELINE.predict_proba(X_df[FEATURE_COLUMNS].values)[:, 1]

    # Map probabilities back to per-observation candidate lists
    candidate_probs_per_obs = []
    meta_idx = 0
    for t in range(len(trip.track)):
        cands = candidates[t]
        k = len(cands)
        p_slice = probs[meta_idx : meta_idx + k].tolist()
        candidate_probs_per_obs.append(p_slice)
        meta_idx += k

    # Classifier-only point predictions (Top-1)
    classifier_point_arcs = [
        candidates[t][int(np.argmax(candidate_probs_per_obs[t]))]["arc_id"]
        for t in range(len(trip.track))
    ]

    # 5. HMM Viterbi Sequence Decoding
    hmm_point_arcs = HMM_MATCHER.decode(trip, candidates, candidate_probs_per_obs)

    # 6. Reconstruct Continuous Route
    reconstructed = reconstruct_continuous_route(trip, hmm_point_arcs)

    # Latency calculation
    latency_ms = (time.time() - t_start) * 1000.0

    # 7. Post-prediction Evaluation (Server-Side Isolated Evaluation)
    eval_metrics = evaluate_predictions_against_ground_truth(
        trip_id=trip_id,
        trip_dir=trip_dir,
        predicted_point_arcs=hmm_point_arcs,
        reconstructed_route_arcs=reconstructed["route_arcs"]
    )

    # Also compute classifier-only accuracy if GT exists
    clf_eval = evaluate_predictions_against_ground_truth(
        trip_id=trip_id,
        trip_dir=trip_dir,
        predicted_point_arcs=classifier_point_arcs
    )
    clf_acc = clf_eval.get("pointwise_accuracy_pct", 0.0) if clf_eval.get("has_ground_truth") else None

    # Prepare response
    raw_track = trip.track.tolist()  # [[lon, lat, ts], ...]

    response_payload = {
        "trip_id": trip_id,
        "partition": partition_name,
        "model_info": MODEL_INFO,
        "num_observations": len(trip.track),
        "latency_ms": round(latency_ms, 2),
        "predicted_point_arcs": hmm_point_arcs,
        "classifier_only_point_arcs": classifier_point_arcs,
        "reconstructed_route": {
            "route_arcs": reconstructed["route_arcs"],
            "route_arc_count": len(reconstructed["route_arcs"]),
            "continuity_pct": reconstructed["continuity_pct"],
            "gaps_found": reconstructed["gaps_found"],
            "gaps_bridged": reconstructed["gaps_bridged"],
            "coordinates": reconstructed["coordinates"],
            "geojson": reconstructed["geojson"]
        },
        "raw_track": raw_track,
        "evaluation": {
            "has_ground_truth": eval_metrics.get("has_ground_truth", False),
            "pointwise_accuracy_pct": eval_metrics.get("pointwise_accuracy_pct"),
            "classifier_accuracy_pct": clf_acc,
            "route_precision": eval_metrics.get("route_precision"),
            "route_recall": eval_metrics.get("route_recall"),
            "route_f1": eval_metrics.get("route_f1"),
            "route_jaccard": eval_metrics.get("route_jaccard")
        }
    }

    # Logging request
    acc_str = f"HMM Acc: {eval_metrics.get('pointwise_accuracy_pct')}% (Clf: {clf_acc}%)" if eval_metrics.get("has_ground_truth") else "GT: N/A"
    logger.info(
        f"Trip: {trip_id} | Points: {len(trip.track)} | Arcs: {len(reconstructed['route_arcs'])} | "
        f"Latency: {latency_ms:.1f}ms | {acc_str} | Continuity: {reconstructed['continuity_pct']}%"
    )

    return response_payload
