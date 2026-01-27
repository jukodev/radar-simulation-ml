# Plausibility Analysis Report

Generated from 19,432 real trajectories with 5,487,505 data points.

## 1. Plausibility Criteria (from Real Data)

| Criterion | Min | Max | P1 | P99 | Mean | Std |
|-----------|-----|-----|----|----|------|-----|
| Jump Distance (NM) | - | 3.1448 | - | 0.6684 | 0.4065 | 0.1425 |
| Speed (kn) | -0.5 | 0.6 | 0.0 | 0.1 | 0.1 | 0.0 |
| Climb Rate (FL/step) | -1169.00 | 1331.00 | -11.00 | 1.00 | -3.7142 | 8.03 |
| Turn Rate (°/step) | - | 180.00 | - | 13.10 | 2.06 | 4.99 |
| Rho (NM) | -1.56 | 109.97 | 2.24 | 99.47 | - | - |
| Flight Level | 0 | 2665 | 0 | 1112 | - | - |
| Speed Change (kn/step) | - | 0.25 | - | 0.01 | 0.0013 | 0.00 |

## 2. Model Evaluation Summary

### Violation Rates (%)

| Model | Jump Dist | Speed | Climb | Turn | Rho | FL | Speed Δ |
|-------|-----------|-------|-------|------|-----|----|---------|
| LSTM h600 l2 d0 | 43.52 | 7.15 | 53.98 | 6.33 | 1.31 | 11.08 | 6.72 |
| LSTM h600 l2 d0.1 | 29.24 | 4.10 | 28.82 | 3.05 | 0.17 | 0.30 | 2.15 |
| LSTM h400 l2 d0 | 35.09 | 10.81 | 42.22 | 4.12 | 1.57 | 7.93 | 5.77 |
| LSTM h400 l3 d0 | 32.50 | 11.38 | 40.26 | 3.52 | 1.30 | 9.16 | 6.26 |
| LSTM h200 l2 d0 | 26.99 | 7.25 | 45.56 | 1.59 | 1.41 | 9.16 | 2.05 |
| MLP h128 | 23.91 | 2.87 | 7.56 | 0.03 | 2.50 | 5.40 | 1.18 |

### Overall Quality

| Model | Plausible % | Diverged % | Avg Violations |
|-------|-------------|------------|----------------|
| LSTM h600 l2 d0 | 0.00 | 0.00 | 259.09 |
| LSTM h600 l2 d0.1 | 0.00 | 0.00 | 135.02 |
| LSTM h400 l2 d0 | 0.00 | 0.00 | 214.15 |
| LSTM h400 l3 d0 | 0.00 | 0.00 | 207.95 |
| LSTM h200 l2 d0 | 0.20 | 0.00 | 187.26 |
| MLP h128 | 0.80 | 3.40 | 86.59 |

## 3. Interpretation

- **Plausible %**: Percentage of trajectories with zero violations
- **Diverged %**: Percentage of trajectories that exceeded the radar range
- **Lower violation rates** indicate better conformance to real flight dynamics
- Limits used are **99th percentile** values from real data
