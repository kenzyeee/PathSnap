"""
Route Reconstruction and Gap Filling Module for PathSnap.

Takes the sequence of per-observation matched arcs, deduplicates consecutive
identical segments, verifies topological continuity, and fills graph shortest-path
gaps between disconnected segments to produce a continuous road-network route.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from src.data.loader import TripData


def reconstruct_continuous_route(
    trip: TripData,
    point_arcs: List[int]
) -> Dict[str, Any]:
    """
    Reconstructs a deduplicated, continuous route sequence from per-observation arc IDs.
    Fills topological gaps using graph shortest path where possible.
    """
    if not point_arcs:
        return {
            "route_arcs": [],
            "continuity_pct": 0.0,
            "gap_count": 0,
            "coordinates": [],
            "geojson": {"type": "FeatureCollection", "features": []}
        }

    # 1. Deduplicate consecutive identical arcs
    deduped_arcs = [point_arcs[0]]
    for arc_id in point_arcs[1:]:
        if arc_id != deduped_arcs[-1]:
            deduped_arcs.append(arc_id)

    # 2. Check and fill gaps between consecutive arcs
    graph = trip.get_graph()
    continuous_route: List[int] = []
    gaps_found = 0
    gaps_bridged = 0

    for i in range(len(deduped_arcs)):
        current_arc = deduped_arcs[i]
        continuous_route.append(current_arc)

        if i < len(deduped_arcs) - 1:
            next_arc = deduped_arcs[i + 1]
            to_node = int(trip.arcs[current_arc, 1])
            from_node = int(trip.arcs[next_arc, 0])

            if to_node != from_node:
                gaps_found += 1
                try:
                    # Find shortest path in road network to bridge the gap
                    sp_nodes = nx.shortest_path(graph, source=to_node, target=from_node, weight="weight")
                    # Convert node sequence to arc sequence
                    for k in range(len(sp_nodes) - 1):
                        u, v = sp_nodes[k], sp_nodes[k + 1]
                        bridge_arc = graph[u][v]["arc_id"]
                        continuous_route.append(bridge_arc)
                    gaps_bridged += 1
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass  # Keep disconnected if no path exists

    # 3. Calculate route continuity
    connected_steps = 0
    total_steps = max(1, len(continuous_route) - 1)
    for i in range(len(continuous_route) - 1):
        a1, a2 = continuous_route[i], continuous_route[i + 1]
        if int(trip.arcs[a1, 1]) == int(trip.arcs[a2, 0]):
            connected_steps += 1

    continuity_pct = (connected_steps / total_steps) * 100.0

    # 4. Extract geographical coordinates [lat, lon] for Leaflet / Folium
    coords_latlon: List[List[float]] = []
    features = []

    for arc_id in continuous_route:
        u = int(trip.arcs[arc_id, 0])
        v = int(trip.arcs[arc_id, 1])
        u_lon, u_lat = float(trip.nodes[u, 0]), float(trip.nodes[u, 1])
        v_lon, v_lat = float(trip.nodes[v, 0]), float(trip.nodes[v, 1])

        coords_latlon.append([u_lat, u_lon])
        coords_latlon.append([v_lat, v_lon])

        # GeoJSON LineString (GeoJSON uses [lon, lat])
        features.append({
            "type": "Feature",
            "properties": {
                "arc_id": int(arc_id),
                "from_node": u,
                "to_node": v,
                "length_m": float(trip.arc_lengths[arc_id])
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[u_lon, u_lat], [v_lon, v_lat]]
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return {
        "route_arcs": continuous_route,
        "continuity_pct": round(continuity_pct, 2),
        "gaps_found": gaps_found,
        "gaps_bridged": gaps_bridged,
        "coordinates": coords_latlon,
        "geojson": geojson
    }
