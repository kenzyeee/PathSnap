# PathSnap Data Quality & Exploratory Analysis Log

**Generated on:** 2026-11-18  
**Scope:** 80 Training Trips, 10 Student-Testing Trips (Total 90 Trips Audited)  
**Strict Compliance:** University ML Hackathon Guidelines (UE24CS352A)

---

## 1. Dataset Inventory & Partition Counts

| Partition | Trip Count | Total GPS Points | Mean Points / Trip | Min Points | Max Points | Mean File Size (KB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 80 | 208417 | 2605.2 | 361 | 19313 | 1095.4 |
| **Student Test** | 10 | 19160 | 1916.0 | 262 | 3958 | 426.6 |
| **Total** | 90 | 227577 | 2528.6 | 262 | 19313 | 1021.1 |

---

## 2. Sampling Rate & Temporal Regularity

- **Nominal Sampling Rate:** 1 Hz (1 sample per second).
- **1s Sampling Compliance:** Across all 90 trips, an average of **94.32%** of consecutive observations are separated by exactly 1.0 second (tolerance ±0.05s).
- **Time Gaps:**
  - Maximum time gap observed in Train: **153064.0 seconds**.
  - Maximum time gap observed in Student Test: **210.0 seconds**.
  - Total gaps exceeding 5 seconds: **1260 occurrences** across all 90 trips.
- **Decision:** Large GPS gaps (>10s) represent potential signal loss or temporary stops. During HMM route decoding, when $\Delta t > 10\text{s}$, transition probabilities must be bridged or reset to avoid penalizing true travel distance across the gap.

---

## 3. Coordinate Ordering & Geographic Verification

- **Format in Raw Files:** `.track` and `.nodes` store 2-3 whitespace-separated columns.
- **Empirical Range:**
  - Column 0 values: `[-100.4185, 137.2370]` -> Corresponds to **Longitude** ($x \in [-180, +180]$).
  - Column 1 values: `[-34.6403, 61.6996]` -> Corresponds to **Latitude** ($y \in [-90, +90]$).
- **Plausibility Check:**
  - Representative coordinates (e.g. Lon ~8.59°E, Lat ~49.85°N) correspond to Darmstadt / Hessen, Germany.
  - Other trips lie in realistic metropolitan networks across North America and Europe.
- **Critical Coordinate Trap Avoided:**
  - Folium and Leaflet require `[latitude, longitude]`. Raw files provide `longitude` in col 0 and `latitude` in col 1.
  - All visualization methods explicitly map `[row[1], row[0]]` to prevent spatial inversion bugs.
- **Metric Projection Decision:**
  - Degree-Euclidean distance is strictly forbidden.
  - Every trip is projected into a local Azimuthal Equidistant coordinate system (`pyproj +proj=aeqd +datum=WGS84`) centered at the trip's geographic midpoint. All candidate generation, distances, speeds, and projections operate in physical meters.

---

## 4. Road Network Characteristics & Connectivity

- **Nodes per Trip:** Mean = 22490.8 (range: 299 to 155003).
- **Arcs per Trip:** Mean = 45132.7 (range: 649 to 307392).
- **Network Sizes:** Files range from ~5 KB up to ~4 MB for metropolitan networks.
- **Average Out-Degree:** 2.02 arcs per node.
- **Isolated Nodes:** 0 isolated nodes detected across all trips (nodes with 0 connected arcs).
- **Directionality:** Arcs represent directed edges from `from_node_id` to `to_node_id`. Ground truth routes strictly follow directed connectivity (`to_node(k) == from_node(k+1)`).

---

## 5. Missing / Invalid Values & Anomalies

- Missing values (NaN, Inf, Null) in `.track`, `.nodes`, `.arcs`: **0 detected**.
- Duplicate timestamps: **0 detected**.
- Timestamp monotonicity: Timestamps are strictly non-decreasing in all 90 trips.
- Zero-length arcs: < 0.05% of arcs have length < 0.1m (handled by clipping to minimum 1.0m to avoid zero-division in graph weights).

---

## 6. Summary of Architectural Decisions

1. **Local Metric Coordinate System:** PyProj AEQD centered on trip mean coordinates for meter-accurate Euclidean distance, bearing, and projection.
2. **Monotonic DP Ground-Truth Alignment:** Maps discrete 100-arc ground-truth routes to ~1000 GPS points with monotonic time constraint.
3. **KDTree Spatial Indexing:** Built over arc midpoints with candidate search radius $R=65\text{m}$ and minimum $K=3$ candidates to ensure Candidate Miss Rate < 1%.
4. **Isolated Evaluation Sandbox:** Ground truth `.route` is never loaded or queried during candidate generation, feature extraction, model inference, or HMM decoding.
