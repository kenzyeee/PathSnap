# PathSnap Bayesian Hyperparameter Optimisation Log

- **Search Strategy:** Optuna Tree-structured Parzen Estimator (TPE)
- **Objective:** Mean Observation-Level Top-1 Accuracy on 5-Fold GroupKFold
- **Trials Completed:** 40
- **Best Trial Number:** 27
- **Best CV Top-1 Accuracy:** 84.61%
- **Total Search Duration:** 841.1 seconds

## Best Hyperparameters

```json
{
  "learning_rate": 0.04111666091908079,
  "max_iter": 200,
  "max_depth": 12,
  "min_samples_leaf": 55,
  "l2_regularization": 0.003653847832923959,
  "max_bins": 255
}
```

## Tuned vs Untuned Performance Comparison

| Model Configuration                     | CV Macro F1-score   | Student-testing Top-1 Accuracy   |   Student-testing Macro F1-score |   Log Loss | Inference Time (ms / trip)   | Training Time   |
|:----------------------------------------|:--------------------|:---------------------------------|---------------------------------:|-----------:|:-----------------------------|:----------------|
| Untuned Baseline (HistGradientBoosting) | N/A (Default)       | 83.70%                           |                           0.9058 |     0.0821 | 13.9 ms                      | 7.09 s          |
| Optuna Tuned (Best Trial)               | 0.8461              | 83.91%                           |                           0.9072 |     0.0781 | 48.7 ms                      | 20.64 s         |
