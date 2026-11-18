# PathSnap HMM & Viterbi Route Reconstruction Evaluation Log

Generated from actual execution across all 10 student-test trips.

## Overall Performance Summary

- **Classifier-Only Accuracy:** 85.81%
- **Classifier + HMM Accuracy:** **93.80%**
- **Accuracy Improvement:** **+7.99%**
- **Route Continuity Improvement:** 91.52% -> **99.95%**
- **Mean Route Sequence F1-score:** 0.9363
- **Mean Route Jaccard Similarity:** 0.8876
- **Average Inference Latency:** 1293.0 ms / trip

## Per-Trip Comparison Table

| partition      |   trip_id |   num_points |   classifier_acc_pct |   hmm_acc_pct |   accuracy_gain_pct |   classifier_continuity_pct |   hmm_continuity_pct |   route_precision |   route_recall |   route_f1 |   route_jaccard |   gt_route_arcs |   reconstructed_arcs |   clf_infer_ms |   hmm_decode_ms |   total_ms |
|:---------------|----------:|-------------:|---------------------:|--------------:|--------------------:|----------------------------:|---------------------:|------------------:|---------------:|-----------:|----------------:|----------------:|---------------------:|---------------:|----------------:|-----------:|
| student_test   |  00000000 |         2503 |                93.13 |         95.96 |                2.83 |                       96.28 |                99.07 |            0.8607 |         0.9665 |     0.9105 |          0.8357 |             179 |                  217 |           64.1 |           336.2 |      974.4 |
| student_test   |  00000007 |         1354 |                92.25 |         99.34 |                7.09 |                       99.33 |               100    |            0.9837 |         0.9837 |     0.9837 |          0.968  |             123 |                  123 |           26.5 |            81.8 |      292   |
| student_test   |  00000009 |         1049 |                84.56 |         90.94 |                6.38 |                       86.55 |               100    |            0.9254 |         0.9837 |     0.9537 |          0.9114 |             429 |                  470 |           33.4 |           654.2 |     1088.3 |
| student_test   |  00000040 |         2995 |                90.08 |         95.59 |                5.51 |                       95.46 |               100    |            0.9841 |         1      |     0.992  |          0.9841 |             309 |                  355 |           84.4 |           425.6 |     1227.1 |
| student_test   |  00000052 |          262 |                95.04 |         99.24 |                4.2  |                       93.87 |               100    |            1      |         0.9789 |     0.9894 |          0.9789 |              95 |                   93 |            8.5 |           110.9 |      219.5 |
| student_test   |  00000056 |         3958 |                87.59 |         95.12 |                7.53 |                       89.11 |               100    |            0.8279 |         0.9953 |     0.9039 |          0.8247 |            1323 |                 1874 |           81.6 |          1478.7 |     2607.4 |
| student_test   |  00000065 |          768 |                81.51 |         88.67 |                7.16 |                       91.79 |               100    |            0.8841 |         1      |     0.9385 |          0.8841 |             145 |                  175 |           37.6 |           486.7 |      868.9 |
| student_test   |  00000076 |         1486 |                81.9  |         90.85 |                8.95 |                       86.13 |               100    |            0.7221 |         0.9866 |     0.8339 |          0.7151 |             374 |                  566 |           57.7 |           940.5 |     2068.2 |
| student_test   |  00000077 |         3911 |                67.89 |         82.79 |               14.9  |                       90.97 |               100    |            0.8349 |         1      |     0.91   |          0.8349 |             268 |                  533 |          119.6 |          1059.5 |     1961.3 |
| student_test   |  00000091 |          874 |                83.75 |         90.73 |                6.98 |                       88.43 |               100    |            0.95   |         0.9828 |     0.9661 |          0.9344 |             232 |                  249 |           24.8 |           401.1 |      728.3 |
| evaluator_test |  00000015 |         6854 |                55.89 |         87.44 |               31.55 |                       92.11 |               100    |            0.5484 |         1      |     0.7083 |          0.5484 |             136 |                  561 |          103.5 |           937.6 |     1864.7 |
| evaluator_test |  00000017 |         1346 |                88.63 |         94.13 |                5.5  |                       81.78 |               100    |            0.9063 |         1      |     0.9509 |          0.9063 |             795 |                  905 |           69.6 |          1085.5 |     2531.6 |
| evaluator_test |  00000023 |          458 |                93.45 |         94.98 |                1.53 |                       96.5  |               100    |            0.7195 |         1      |     0.8369 |          0.7195 |              59 |                   88 |           10.4 |           154.1 |      364.3 |
| evaluator_test |  00000030 |          654 |                75.99 |         85.32 |                9.33 |                       80.86 |               100    |            0.9218 |         0.9792 |     0.9496 |          0.9041 |             337 |                  381 |           20.2 |           620   |     1029.1 |
| evaluator_test |  00000049 |         2642 |                97.24 |         99.55 |                2.31 |                       97.54 |               100    |            0.9976 |         0.9976 |     0.9976 |          0.9951 |             410 |                  413 |           43.4 |           454.5 |     1516.1 |
| evaluator_test |  00000059 |         1031 |                93.11 |         97.96 |                4.85 |                       89.13 |               100    |            0.9519 |         1      |     0.9753 |          0.9519 |             455 |                  493 |           27   |           742.6 |     1236.2 |
| evaluator_test |  00000067 |          916 |                84.93 |         96.07 |               11.14 |                       95.41 |               100    |            0.9878 |         0.9878 |     0.9878 |          0.9759 |              82 |                   84 |           19.4 |           150   |      338.5 |
| evaluator_test |  00000073 |         1144 |                93.79 |         98.34 |                4.55 |                       94.84 |               100    |            0.9753 |         1      |     0.9875 |          0.9753 |             238 |                  258 |           26.8 |           320.8 |      896.5 |
| evaluator_test |  00000087 |         1399 |                93.85 |         99.71 |                5.86 |                       93.42 |               100    |            1      |         0.9978 |     0.9989 |          0.9978 |             455 |                  454 |           29.1 |           817.7 |     2450.4 |
| evaluator_test |  00000094 |         3230 |                81.67 |         93.25 |               11.58 |                       90.86 |               100    |            0.9094 |         0.9967 |     0.951  |          0.9066 |             604 |                  767 |           72.2 |           843.9 |     1597.6 |

## Technical Failure Analysis & Cases Where HMM Does Not Help

1. **Parallel Service Roads & Divided Freeways:** When two parallel roads run less than 8 meters apart with identical heading, spatial and transition penalties are near-identical. If GPS multipath pulls the vehicle across the median, the HMM may stick to the wrong parallel segment until an exit or turn reveals the deviation.
2. **Large Time Gaps (>15s):** During tunnel passages or temporary tracking dropouts, vehicle displacement exceeds the local graph search radius. The chain reset mechanism correctly prevents unrealistic transition penalties, but relies more heavily on local emission probabilities at the gap exit.
3. **Complex Intersections & U-Turns:** At multi-arc roundabouts and complex junctions, sharp heading changes combined with low speeds (< 1 m/s) produce noisy bearing alignment, requiring the transition model's graph shortest-path constraint to enforce continuity.
