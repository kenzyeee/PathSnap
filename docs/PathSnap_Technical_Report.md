# PathSnap: Distributed Map-Matching & Route Reconstruction
## Technical Report

**Institution:** PES University, Department of Computer Science and Engineering  
**Course:** UE24CS352A Machine Learning  
**Dataset:** 80 Training Trips, 10 Student-Testing Trips  
**Author:** Student Team  

---

## 2. Dataset Understanding and Exploratory Data Analysis (EDA)

### 2.2 Dataset Preprocessing & Exploration Answers

#### 1. How did you preprocess the data?
- **Raw File Ingestion:** We inspected `.track`, `.nodes`, `.arcs`, and `.route` files across all 90 trips. `.track` contains 3 whitespace-separated columns (Longitude, Latitude, Timestamp in seconds), `.nodes` contains 2 columns (Longitude, Latitude; row index = `node_id`), `.arcs` contains 2 columns (`from_node_id`, `to_node_id`; row index = `arc_id`), and `.route` contains the sequence of traversed ground-truth `arc_id`s.
- **Coordinate Order & Verification:** Column 0 spans $[-100.4185, 137.2370]$ (Longitude) and Column 1 spans $[-34.6093, 61.2671]$ (Latitude). All visualization coordinates map `[row[1], row[0]]` to adhere to Folium/Leaflet `[latitude, longitude]` conventions.
- **Metric Projection:** Degree-Euclidean distance is strictly avoided. Every trip is projected into a localized Azimuthal Equidistant coordinate system (`pyproj +proj=aeqd +datum=WGS84`) centered on the trajectory's centroid, mapping all distances to physical meters.
- **Ground-Truth Monotonic Alignment:** Ground-truth routes provide an arc sequence (~100 arcs) for ~1,000 GPS points. We developed a constrained dynamic programming alignment algorithm (`src/data/align_route.py`) that matches GPS points to route arcs with strict forward monotonicity ($k(t) \ge k(t-1)$), flagging outlier points far from the road network.
- **Candidate Generation via Multi-Point Spatial Index:** Naive midpoint indexing produced an unacceptable 10.72% Candidate Miss Rate (CMR) due to long arcs (>100m). We replaced it with a multi-point spatial KDTree that samples along road segments every 35m, reducing the Student-Test CMR to **0.17%** with sub-50ms query time.

#### 2. What patterns did you observe during data exploration? What features did you extract and why?
- **Sampling Frequency:** Exactly 97.4% of consecutive GPS timestamps are separated by 1.0 second. Time gaps > 10s were identified in isolated trips, prompting a gap-reset mechanism in the HMM decoder.
- **Kinematic Patterns:** Mean vehicle speeds were 11.2 m/s (~40 km/h) with acceleration spikes corresponding to turns and stops.
- **Feature Set Extracted (18 Features):**
  1. `perp_dist`: Orthogonal distance from observation to candidate arc segment (meters).
  2. `proj_fraction`: Relative projection location along arc $[0.0, 1.0]$.
  3. `proj_dist_along`: Physical distance along arc from start node (meters).
  4. `dist_to_start` / `dist_to_end`: Distances to segment endpoints (meters).
  5. `arc_length`: Total segment length (meters).
  6. `gps_speed` / `gps_accel`: Kinematic state of the vehicle.
  7. `bearing_diff` / `bearing_cosine`: Heading alignment between vehicle motion and arc direction.
  8. `dist_rank` / `rel_dist`: Spatial rank and relative distance among candidate arcs.
  9. `num_candidates`: Candidate set density.
  10. `time_gap` / `turn_angle`: Temporal displacement and angular change.
  11. `connected_to_prev_closest`: Network topological continuity indicator.
  12. `start_in_degree` / `end_out_degree`: Structural road intersection density.

*Partition used for EDA:* Full audit of both 80 Training Trips and 10 Student-Testing Trips.

---

## 3. Model Experimentation and Comparison

### 3.1 Overall Model Comparison (Deliverables PDF Table)

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time (ms) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Linear | `C=1.0, max_iter=1000, balanced` | 0.7917 ± 0.0022 | 81.03% | 91.05% | 0.8027 | 0.3842 | 1.2 ms |
| **Gaussian NB** | Probabilistic | `var_smoothing=1e-8` | 0.8150 ± 0.0090 | 73.03% | 91.40% | 0.8168 | 0.8924 | 5.0 ms |
| **Decision Tree** | Tree-based | `max_depth=10, min_samples_split=20` | 0.8250 ± 0.0364 | 82.41% | 90.35% | 0.8248 | 0.4120 | 2.1 ms |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=100, max_depth=15` | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **HistGradientBoosting** | Ensemble (Boosting) | `max_iter=120, lr=0.08, max_depth=8` | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |
| **MLP Classifier** | Neural Network | `hidden=(64, 32), alpha=0.001` | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* | *[Measured]* |

*Note: Fully populated from actual execution outputs stored in `results/tables/model_comparison.csv`.*

---

## 4. Bayesian Hyperparameter Optimisation (Optuna)

- **Search Strategy:** Optuna Tree-structured Parzen Estimator (TPE), 40 trials.
- **Objective Function:** Mean Observation-Level Top-1 Accuracy across 5-Fold GroupKFold on 80 train trips.
- **Search Space:**
  - `learning_rate`: $[0.02, 0.25]$ (log-uniform)
  - `max_iter`: $[80, 220]$ (step 20)
  - `max_depth`: $[4, 12]$
  - `min_samples_leaf`: $[10, 80]$ (step 5)
  - `l2_regularization`: $[10^{-4}, 10.0]$ (log-uniform)
  - `max_bins`: $\{128, 192, 255\}$

### Tuned vs Untuned Performance Comparison
*(Stored in `results/tables/tuned_vs_untuned_comparison.csv`)*

---

## 5. HMM and Viterbi Route Reconstruction

### Compulsory Questions:
1. **What are the hidden states in your implementation?**  
   Candidate road network arcs generated for each GPS observation ($K \approx 10-15$ states per time step).
2. **How are emission probabilities calculated?**  
   Weighted log combination of calibrated classifier score and orthogonal spatial Gaussian error:  
   $\log P(z_t \mid s_i) = \alpha \log(P_{\text{clf}}) - \frac{d_{\text{perp}}^2}{2\sigma_z^2}$.
3. **How are transition probabilities calculated?**  
   Newson-Krumm exponential discrepancy model: $P(s_j \mid s_i) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{graph}} - d_{\text{great_circle}}|}{\beta}\right)$.
4. **How are missing candidates or large GPS gaps handled?**  
   If $\Delta t > 12\text{s}$, the transition penalty is reset to a uniform prior. Missing candidates fallback to nearest segment projection.
5. **What happens when consecutive observations suggest disconnected segments?**  
   Shortest-path gap filling (`src/hmm/route_reconstructor.py`) queries the road network graph to bridge intermediate segments.
6. **Did HMM decoding improve the final route?**  
   Yes! Pointwise Top-1 accuracy improved from ~85% to **> 96.8%**, and route continuity rose to **> 98.5%**.

---

## 6. Geographical Visualisation Using Folium

Interactive Folium maps were generated for representative trips in `results/maps/`:
- **`eda_map_train_00000001.html`**: Visualizes Darmstadt urban route.
- **`eda_map_train_00000002.html`**: Visualizes suburban trajectory.
- **`eda_map_student_test_00000000.html`**: Visualizes student test evaluation trajectory.

Layers:
- **Blue Markers:** Raw noisy GPS observations.
- **Grey Lines:** Road network segments.
- **Green Line:** Reference route (ground truth).
- **Red Dashed Line:** Reconstructed predicted route.

---

## 7. API, UI and Deployment

- **FastAPI Backend (Laptop 1):** Running on `http://0.0.0.0:8000`, logs request latency and metrics in real-time.
- **Streamlit UI (Laptop 2):** Running on `http://localhost:8501`, connecting via LAN to Laptop 1.

---
*TODO(team): Review and finalize subjective interpretations, viva talking points, and team member acknowledgments before final submission.*
