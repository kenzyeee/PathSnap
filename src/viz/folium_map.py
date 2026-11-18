"""
Folium Map Visualisation Module for PathSnap.

Generates interactive multi-layer Leaflet maps using Folium:
- Raw GPS points (blue circles with tooltips)
- Road network geometry (light grey lines)
- Ground-truth reference route (bold green line)
- Predicted route (bold red line, when available)
- Custom legend and LayerControl
"""

import os
import folium
from typing import List, Tuple, Dict, Any, Optional


def create_trip_map(
    trip_id: str,
    track_points: List[Tuple[float, float, float]],  # (lon, lat, time)
    nodes: Dict[int, Tuple[float, float]],           # id -> (lon, lat)
    arcs: List[Tuple[int, int]],                     # (from_id, to_id)
    reference_route_arcs: Optional[List[int]] = None,
    predicted_route_arcs: Optional[List[int]] = None,
    include_network: bool = True,
    max_network_arcs: int = 5000,
    output_html_path: Optional[str] = None
) -> folium.Map:
    """
    Creates an interactive Folium map for a single trip.
    Note: Track coordinates are stored as (lon, lat), while Folium expects [lat, lon].
    """
    if not track_points:
        raise ValueError(f"No track points provided for trip {trip_id}")

    # Center map on average lat/lon of track
    avg_lat = sum(p[1] for p in track_points) / len(track_points)
    avg_lon = sum(p[0] for p in track_points) / len(track_points)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # 1. Road Network Layer (optional or limited for large networks)
    if include_network and arcs:
        fg_network = folium.FeatureGroup(name="Road Network", show=False)
        step = max(1, len(arcs) // max_network_arcs)
        for i in range(0, len(arcs), step):
            u, v = arcs[i]
            if u in nodes and v in nodes:
                u_lon, u_lat = nodes[u]
                v_lon, v_lat = nodes[v]
                folium.PolyLine(
                    locations=[[u_lat, u_lon], [v_lat, v_lon]],
                    color="#888888",
                    weight=1.5,
                    opacity=0.45,
                    tooltip=f"Arc {i}: {u}->{v}"
                ).add_to(fg_network)
        fg_network.add_to(m)

    # 2. Reference Route (Ground Truth - Green)
    if reference_route_arcs:
        fg_ref = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        ref_coords = []
        for arc_id in reference_route_arcs:
            if 0 <= arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u in nodes and v in nodes:
                    u_lon, u_lat = nodes[u]
                    v_lon, v_lat = nodes[v]
                    ref_coords.append([[u_lat, u_lon], [v_lat, v_lon]])
        
        for segment in ref_coords:
            folium.PolyLine(
                locations=segment,
                color="#28a745",  # Green
                weight=4.5,
                opacity=0.85
            ).add_to(fg_ref)
        fg_ref.add_to(m)

    # 3. Predicted Route (Red, if provided)
    if predicted_route_arcs:
        fg_pred = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
        pred_coords = []
        for arc_id in predicted_route_arcs:
            if 0 <= arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u in nodes and v in nodes:
                    u_lon, u_lat = nodes[u]
                    v_lon, v_lat = nodes[v]
                    pred_coords.append([[u_lat, u_lon], [v_lat, v_lon]])
        
        for segment in pred_coords:
            folium.PolyLine(
                locations=segment,
                color="#dc3545",  # Red
                weight=4.0,
                opacity=0.9,
                dash_array="6, 6"
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    # 4. Raw GPS Observations (Blue dots)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    for idx, (lon, lat, ts) in enumerate(track_points):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#0066cc",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.7,
            weight=1,
            tooltip=f"Pt {idx} | t={ts:.1f}s | ({lat:.5f}, {lon:.5f})"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Start and End Markers
    start_lon, start_lat, _ = track_points[0]
    end_lon, end_lat, _ = track_points[-1]
    folium.Marker(
        [start_lat, start_lon],
        popup=f"Start: Trip {trip_id}",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)
    folium.Marker(
        [end_lat, end_lon],
        popup=f"End: Trip {trip_id}",
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 25px; left: 25px; width: 220px; height: 160px; 
                background-color: white; z-index:9999; font-size:13px;
                border:2px solid #ccc; border-radius: 8px; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b>PathSnap Map Legend</b><br>
        <span style="color:#0066cc;">●</span> Raw GPS Points<br>
        <span style="color:#888888;">―</span> Road Network<br>
        <span style="color:#28a745; font-weight:bold;">―</span> Reference Route (GT)<br>
        <span style="color:#dc3545; font-weight:bold;">---</span> Predicted Route<br>
        <hr style="margin: 5px 0;">
        <small>Trip ID: <b>{trip_id}</b> ({len(track_points)} pts)</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)

    return m
