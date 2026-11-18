# PathSnap LambdaRank & Lambda Optimization Report

## 1. Executive Summary
- **Algorithm**: LightGBM LambdaRank / LambdaMART (`LGBMRanker`)
- **Objective Function**: `lambdarank` (Direct optimization of pairwise/listwise NDCG@1)
- **Features Used**: 18 geometric, kinematic, and topological features
- **Training Set**: 805,496 candidate rows across 92,631 observation queries
- **Optimized Lambda Parameter**: **$\lambda^* = 0.60$**
- **Student-Test HMM Accuracy**: **91.44%**
- **Student-Test Continuity**: **100.00%**
- **Mean Latency per Trip**: **1308.2 ms**

## 2. Lambda Optimization Results
| Lambda ($\lambda$) | Validation HMM Accuracy (%) | Min Trip Accuracy (%) | Max Trip Accuracy (%) |
|---|---|---|---|
| 0.20 | 83.73% | 9.00% | 99.64% |
| 0.40 | 84.44% | 9.03% | 99.80% |
| 0.60 **(OPTIMAL)** | 84.80% | 9.05% | 99.80% |
| 0.80 | 84.37% | 8.96% | 99.80% |
| 1.00 | 84.25% | 8.99% | 99.80% |
| 1.20 | 84.50% | 8.98% | 99.80% |
| 1.50 | 84.35% | 8.99% | 99.84% |
| 1.80 | 84.40% | 9.01% | 99.84% |
| 2.00 | 84.37% | 9.06% | 99.84% |
| 2.20 | 84.32% | 9.09% | 99.84% |
| 2.50 | 84.16% | 9.06% | 99.84% |
| 2.80 | 83.78% | 9.05% | 99.84% |
| 3.00 | 83.49% | 9.10% | 99.84% |
| 3.50 | 83.59% | 9.10% | 99.84% |
| 4.00 | 83.42% | 9.11% | 99.84% |
| 5.00 | 83.31% | 9.09% | 99.80% |

## 3. Student-Test Evaluation (10 Trips)
| Trip ID | Observations | Ranker Top-1 Acc (%) | LambdaRank + HMM Acc (%) | Gain (%) | Continuity (%) | Route F1 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| 00000000 | 2503 | 93.29% | 96.80% | +3.51% | 100.0% | 0.9301 | 854.0 |
| 00000007 | 1354 | 87.81% | 87.15% | -0.66% | 100.0% | 0.9798 | 269.5 |
| 00000009 | 1049 | 84.94% | 90.09% | +5.15% | 100.0% | 0.9559 | 916.6 |
| 00000040 | 2995 | 90.45% | 95.86% | +5.41% | 100.0% | 0.9920 | 2214.9 |
| 00000052 | 262 | 95.04% | 99.62% | +4.58% | 100.0% | 0.9894 | 223.3 |
| 00000056 | 3958 | 87.70% | 95.40% | +7.70% | 100.0% | 0.9247 | 2330.9 |
| 00000065 | 768 | 78.91% | 89.32% | +10.41% | 100.0% | 0.9385 | 1301.6 |
| 00000076 | 1486 | 83.31% | 91.39% | +8.08% | 100.0% | 0.8528 | 1909.9 |
| 00000077 | 3911 | 64.82% | 78.14% | +13.32% | 100.0% | 0.9257 | 2339.5 |
| 00000091 | 874 | 83.64% | 90.62% | +6.98% | 100.0% | 0.9744 | 721.6 |

**Summary Averages**:
- Mean Ranker Top-1 Accuracy: **84.99%**
- Mean LambdaRank + HMM Accuracy: **91.44%**
- Mean Route Continuity: **100.00%**
- Mean F1-Score: **0.9463**
- Mean Latency: **1308.2 ms**
