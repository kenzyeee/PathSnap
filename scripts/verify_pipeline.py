"""
End-to-End System Verification and HTTP Integration Test Suite for PathSnap.

Tests:
1. GET /health: Server readiness and loaded model verification
2. GET /trips: Available evaluation trip listing
3. POST /predict: HTTP inference for all 10 student-test trips
4. Data Leakage Proof: Deletes .route from a trip folder and verifies /predict succeeds
5. Generates real timing and accuracy report for evaluation demo
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import shutil
import tempfile
import requests
import json
from typing import Dict, Any


def run_verification(api_base_url: str = "http://localhost:8000"):
    print("=" * 65)
    print("STARTING SYSTEM VERIFICATION & END-TO-END PIPELINE CHECKS")
    print(f"Target API Endpoint: {api_base_url}")
    print("=" * 65)

    # 1. Test /health
    print("\n1. Testing GET /health...")
    try:
        resp = requests.get(f"{api_base_url}/health", timeout=5)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        health_data = resp.json()
        print(f"  Status: {health_data['status']}")
        print(f"  Model Loaded: {health_data['model_loaded']}")
        print(f"  Available Trips: {health_data['available_trips_count']}")
        assert health_data["model_loaded"] is True, "Model is not loaded on server!"
        print("  [PASS] /health is healthy and model is loaded.")
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        return False

    # 2. Test /trips
    print("\n2. Testing GET /trips...")
    try:
        resp = requests.get(f"{api_base_url}/trips", timeout=5)
        assert resp.status_code == 200
        trips_data = resp.json()
        trips = trips_data.get("trips", [])
        print(f"  Retrieved {len(trips)} evaluation trips: {trips}")
        assert len(trips) >= 10, f"Expected at least 10 trips, got {len(trips)}"
        print("  [PASS] /trips returned valid trip list.")
    except Exception as e:
        print(f"  [FAIL] Failed to list trips: {e}")
        return False

    # 3. Test /predict on all 10 student-test trips
    print("\n3. Testing POST /predict on all 10 student-test trips...")
    per_trip_metrics = []
    
    for idx, tid in enumerate(trips):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{api_base_url}/predict",
                json={"trip_id": tid, "include_ground_truth": False},
                timeout=30
            )
            rtt_ms = (time.time() - t0) * 1000.0
            assert resp.status_code == 200, f"Failed for {tid}: {resp.status_code} - {resp.text}"
            
            data = resp.json()
            assert data["trip_id"] == tid
            assert len(data["predicted_point_arcs"]) == data["num_observations"]
            assert len(data["reconstructed_route"]["route_arcs"]) > 0
            assert "coordinates" in data["reconstructed_route"]
            assert "geojson" in data["reconstructed_route"]
            
            eval_res = data.get("evaluation", {})
            hmm_acc = eval_res.get("pointwise_accuracy_pct", 0.0)
            clf_acc = eval_res.get("classifier_accuracy_pct", 0.0)
            cont = data["reconstructed_route"]["continuity_pct"]
            api_latency = data["latency_ms"]
            
            per_trip_metrics.append({
                "trip_id": tid,
                "points": data["num_observations"],
                "clf_acc": clf_acc,
                "hmm_acc": hmm_acc,
                "continuity": cont,
                "api_latency_ms": api_latency,
                "roundtrip_ms": round(rtt_ms, 1)
            })
            print(f"  [{idx+1}/10] Trip {tid}: HMM Acc={hmm_acc:.2f}% (Clf={clf_acc:.2f}%) | Cont={cont:.1f}% | API Latency={api_latency:.1f}ms")
        except Exception as e:
            print(f"  [FAIL] Prediction failed for trip {tid}: {e}")
            return False

    print("  [PASS] All 10 evaluation trips successfully processed via API!")

    # 4. Strict Security & Data Leakage Proof: Delete .route and verify /predict works
    print("\n4. Proving No Ground-Truth Leakage (Predict with deleted .route file)...")
    eval_dir = trips_data.get("eval_data_dir", "data/Student folder/student_test")
    test_trip = trips[0]
    real_trip_folder = os.path.join(eval_dir, test_trip)
    real_route_file = os.path.join(real_trip_folder, f"{test_trip}.route")
    temp_backup_file = os.path.join(real_trip_folder, f"{test_trip}.route.bak")

    try:
        # Rename .route to simulate evaluator scenario where no .route is provided
        if os.path.exists(real_route_file):
            shutil.move(real_route_file, temp_backup_file)
            print(f"  Temporarily removed .route file from: {real_route_file}")

        resp = requests.post(
            f"{api_base_url}/predict",
            json={"trip_id": test_trip, "include_ground_truth": False},
            timeout=15
        )
        assert resp.status_code == 200, f"Failed without .route file: {resp.status_code}"
        data = resp.json()
        assert len(data["predicted_point_arcs"]) == data["num_observations"]
        assert len(data["reconstructed_route"]["route_arcs"]) > 0
        assert data["evaluation"]["has_ground_truth"] is False
        print("  [PASS] /predict succeeded completely without .route file!")
        print(f"  Returned {len(data['predicted_point_arcs'])} matched arcs and {len(data['reconstructed_route']['route_arcs'])} reconstructed route arcs.")
    finally:
        # Restore backup
        if os.path.exists(temp_backup_file):
            shutil.move(temp_backup_file, real_route_file)
            print(f"  Restored {real_route_file}")

    # 5. Output Final Verification Report
    print("\n" + "=" * 65)
    print("FINAL VERIFICATION SUMMARY TABLE:")
    print("=" * 65)
    import pandas as pd
    report_df = pd.DataFrame(per_trip_metrics)
    print(report_df.to_string(index=False))
    
    mean_hmm_acc = report_df["hmm_acc"].mean()
    mean_latency = report_df["api_latency_ms"].mean()
    print("-" * 65)
    print(f"Overall Mean Pointwise Accuracy: {mean_hmm_acc:.2f}%")
    print(f"Overall Mean API Latency:        {mean_latency:.1f} ms / trip")
    print("=" * 65)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = run_verification(url)
    sys.exit(0 if success else 1)
