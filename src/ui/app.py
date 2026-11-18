"""
PathSnap Evaluator User Interface (Runs on Laptop 2).

Streamlit + Folium dashboard for hackathon evaluation:
- Connects to Laptop 1 FastAPI backend (configurable base URL)
- Displays evaluation trips as selectable cards
- Interactive Folium map rendering raw GPS, road network, and predicted route
- Comprehensive performance metrics: Classifier vs HMM accuracy, continuity, latency
"""

import os
import time
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium


st.set_page_config(
    page_title="PathSnap | Map-Matching Evaluator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .success-badge {
        color: #059669;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def get_api_trips(api_url: str):
    """Fetches list of available evaluation trips from the API."""
    try:
        resp = requests.get(f"{api_url}/trips", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("trips", [])
        return []
    except Exception:
        return []


def check_api_health(api_url: str):
    """Checks API health and model status."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


# Sidebar: Connection & Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/route.png", width=64)
    st.title("PathSnap System")
    st.caption("PES University | UE24CS352A Machine Learning")
    st.markdown("---")
    
    st.subheader("⚙️ Deployment Settings")
    default_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_url = st.text_input("Backend API Base URL (Laptop 1):", value=default_url)
    
    health = check_api_health(api_url)
    if health:
        st.success(f"Connected to API ({health['status'].upper()})")
        if health.get("model_loaded"):
            m_name = health.get("model_name", "LambdaRank (Optimized LambdaMART)")
            algo = health.get("algorithm", "Learning-to-Rank (LTR)")
            opt_lambda = health.get("optimized_lambda", 0.60)
            st.info(f"🎯 **Active Model:** `{m_name}`\n\n- **Family:** {algo}\n- **Optimized $\\lambda$:** `{opt_lambda}`\n- **Objective:** NDCG@1")
        else:
            st.warning("Model Status: Not loaded yet")
    else:
        st.error(f"Cannot reach API at: `{api_url}`")
        st.caption("Ensure Laptop 1 is running `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`")
        
    st.markdown("---")
    st.markdown("""
    **Quick Guide:**
    1. Select an evaluation trip from the main panel.
    2. Click **Run PathSnap**.
    3. View the map and performance metrics.
    """)


# Main Content Area
st.markdown('<div class="main-header">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluator Interface (Laptop 2) connecting to Server Model & Decoder (Laptop 1)</div>', unsafe_allow_html=True)

# Default partitions so BOTH options ALWAYS appear
DEFAULT_PARTITIONS = {
    "evaluator_test": ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"],
    "student_test": ["00000000", "00000007", "00000009", "00000040", "00000052", "00000056", "00000065", "00000076", "00000077", "00000091"]
}

available_partitions = DEFAULT_PARTITIONS.copy()
if health and health.get("partitions"):
    for part_name, tids in health["partitions"].items():
        if tids:
            available_partitions[part_name] = tids

# Trip Selection Row
col_part, col_sel, col_btn = st.columns([1.5, 2.5, 1])
with col_part:
    selected_part = st.selectbox(
        "Evaluation Partition:",
        options=list(available_partitions.keys()),
        index=0,
        help="Choose between evaluator_test (new evaluation set) and student_test."
    )
with col_sel:
    trip_options = available_partitions.get(selected_part, [])
    selected_trip = st.selectbox(
        "Select Evaluation Trip:",
        options=trip_options,
        index=0,
        help="Select any trip from the chosen partition."
    )
with col_btn:
    st.write("") # spacing
    st.write("")
    run_btn = st.button("🚀 Run PathSnap", type="primary", use_container_width=True)

# Prediction Execution
if run_btn:
    with st.spinner(f"Sending inference request for trip {selected_trip} to {api_url}..."):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_trip, "include_ground_truth": False},
                timeout=30
            )
            roundtrip_ms = (time.time() - t0) * 1000.0
            
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["result_data"] = data
                st.session_state["roundtrip_ms"] = roundtrip_ms
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API: {str(e)}")

# Display Results if available
if "result_data" in st.session_state and st.session_state["result_data"].get("trip_id") == selected_trip:
    data = st.session_state["result_data"]
    roundtrip = st.session_state.get("roundtrip_ms", 0.0)
    
    st.markdown("---")
    part_label = data.get("partition", selected_part)
    m_info = data.get("model_info", {})
    m_name = m_info.get("model_name", "LambdaRank (Optimized LambdaMART)")
    opt_lam = m_info.get("optimized_lambda", 0.60)
    st.subheader(f"📊 Results for Trip: `{data['trip_id']}` ({part_label})")
    st.caption(f"🚀 **Model Pipeline:** {m_name} | **Calibrated $\\lambda$:** `{opt_lam}` | **Objective:** NDCG@1 Ranking")
    
    # Key Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    
    eval_info = data.get("evaluation", {})
    has_gt = eval_info.get("has_ground_truth", False)
    
    with m1:
        if has_gt:
            st.metric(
                label="HMM Top-1 Accuracy",
                value=f"{eval_info.get('pointwise_accuracy_pct', 0.0):.2f}%",
                delta=f"+{eval_info.get('pointwise_accuracy_pct', 0.0) - (eval_info.get('classifier_accuracy_pct') or 0.0):.2f}% vs Clf" if eval_info.get('classifier_accuracy_pct') else None
            )
        else:
            st.metric(label="Total Observations", value=data.get("num_observations"))
            
    with m2:
        if has_gt and eval_info.get("classifier_accuracy_pct") is not None:
            st.metric(label="Classifier Accuracy", value=f"{eval_info.get('classifier_accuracy_pct'):.2f}%")
        else:
            st.metric(label="Route Arcs", value=data["reconstructed_route"]["route_arc_count"])
            
    with m3:
        continuity = data["reconstructed_route"].get("continuity_pct", 0.0)
        st.metric(label="Route Continuity", value=f"{continuity:.1f}%")
        
    with m4:
        st.metric(label="API Compute Latency", value=f"{data.get('latency_ms', 0):.1f} ms")
        
    with m5:
        st.metric(label="Network Roundtrip", value=f"{roundtrip:.1f} ms")
        
    # Map & Route Visualization
    st.markdown("### 🗺️ Trajectory & Reconstructed Route Map")
    
    raw_track = data.get("raw_track", [])  # [[lon, lat, ts], ...]
    route_coords = data["reconstructed_route"].get("coordinates", [])  # [[lat, lon], ...]
    
    if raw_track:
        avg_lat = sum(p[1] for p in raw_track) / len(raw_track)
        avg_lon = sum(p[0] for p in raw_track) / len(raw_track)
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="OpenStreetMap")
        
        # 1. Raw GPS Observations (Blue dots)
        fg_gps = folium.FeatureGroup(name="Raw GPS Points (Blue)", show=True)
        for idx, p in enumerate(raw_track):
            folium.CircleMarker(
                location=[p[1], p[0]],
                radius=3.5,
                color="#0066cc",
                fill=True,
                fill_color="#3399ff",
                fill_opacity=0.75,
                tooltip=f"Obs {idx}: ({p[1]:.5f}, {p[0]:.5f}) t={p[2]:.1f}s"
            ).add_to(fg_gps)
        fg_gps.add_to(m)
        
        # 2. Reconstructed Route (Red PolyLine)
        if route_coords:
            fg_route = folium.FeatureGroup(name="Predicted Reconstructed Route (Red)", show=True)
            folium.PolyLine(
                locations=route_coords,
                color="#dc3545",
                weight=4.5,
                opacity=0.9,
                tooltip=f"Trip {data['trip_id']} Reconstructed Route"
            ).add_to(fg_route)
            fg_route.add_to(m)
            
        # Start and Finish pins
        folium.Marker([raw_track[0][1], raw_track[0][0]], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker([raw_track[-1][1], raw_track[-1][0]], popup="Finish", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=550)
        
    # Additional Details Accordion
    with st.expander("🔍 View Technical Route Details & GeoJSON"):
        st.write(f"**Route Arc Sequence ({len(data['reconstructed_route']['route_arcs'])} segments):**")
        st.write(data["reconstructed_route"]["route_arcs"][:50], "... (truncated)" if len(data["reconstructed_route"]["route_arcs"]) > 50 else "")
        if has_gt:
            st.write(f"**Route Precision:** {eval_info.get('route_precision')} | **Recall:** {eval_info.get('route_recall')} | **F1:** {eval_info.get('route_f1')}")
        st.json(data["reconstructed_route"]["geojson"]["features"][:3])
