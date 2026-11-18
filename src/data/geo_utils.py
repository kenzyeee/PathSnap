"""
Geographical & Spatial Utilities Module for PathSnap.

Strictly adheres to guideline: "Never use degree-Euclidean distance. 
Use a local metric projection (pyproj, centred on each trip) or geodesic distance."
"""

import math
import numpy as np
import pyproj
from typing import Tuple, List, Optional


class LocalProjector:
    """
    Projector that centers an Azimuthal Equidistant projection on the trip center.
    Preserves true distances (in meters) from the center in all directions.
    """
    def __init__(self, center_lon: float, center_lat: float):
        self.center_lon = center_lon
        self.center_lat = center_lat
        proj_str = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        self.transformer_fwd = pyproj.Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
        self.transformer_inv = pyproj.Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)

    def to_meters(self, lon: float, lat: float) -> Tuple[float, float]:
        """Converts (lon, lat) in degrees to (x, y) in meters."""
        x, y = self.transformer_fwd.transform(lon, lat)
        return float(x), float(y)

    def to_meters_batch(self, lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts numpy arrays of (lons, lats) to arrays of (x, y) in meters."""
        xs, ys = self.transformer_fwd.transform(lons, lats)
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    def to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """Converts (x, y) in meters back to (lon, lat) in degrees."""
        lon, lat = self.transformer_inv.transform(x, y)
        return float(lon), float(lat)


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculates compass bearing from point 1 to point 2 in degrees [0, 360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def angle_diff(b1: float, b2: float) -> float:
    """Calculates absolute angular difference between two bearings in degrees [0, 180]."""
    diff = abs(b1 - b2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def point_to_segment_projection_2d(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, float, float, float]:
    """
    Projects point P(px, py) onto line segment AB: A(ax, ay) to B(bx, by) in Cartesian meters.
    Returns:
        (perp_dist, proj_u_clamped, proj_x, proj_y)
        - perp_dist: distance from P to the closest point on segment AB
        - proj_u_clamped: fraction along segment [0.0, 1.0]
        - (proj_x, proj_y): coordinates of closest point
    """
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx * vx + vy * vy
    
    if seg_len_sq < 1e-9:
        # Segment is degenerate (point)
        dist = math.hypot(px - ax, py - ay)
        return dist, 0.0, ax, ay
        
    wx = px - ax
    wy = py - ay
    u = (wx * vx + wy * vy) / seg_len_sq
    u_clamped = max(0.0, min(1.0, u))
    
    proj_x = ax + u_clamped * vx
    proj_y = ay + u_clamped * vy
    perp_dist = math.hypot(px - proj_x, py - proj_y)
    
    return perp_dist, u_clamped, proj_x, proj_y
