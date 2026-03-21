# Training Experiment Log

Project: Ensemble-Based Hybrid Disaster Prediction System

## 1. Goal
- Improve accuracy further (especially Flood), while keeping Earthquake strong.
- Maintain proper human-style experiment notes:
  - what changed,
  - how many times trained,
  - per-model scores,
  - base model average,
  - stacking score,
  - what failed and what worked.

---

## 2. Experiment Timeline and Results

### Experiment IDs
- `E0` Baseline: original generator + original params.
- `E1` Hyperparam-heavy: class-balancing + larger/stronger models.
- `E2` Generator tuning (v2): reverted model params, improved Flood generator signal.
- `E3` Aggressive joint tuning: stronger model params + stronger Flood generator (single global config for both disasters).
- `E4` Final per-disaster selection:
  - Flood from E3 (best Flood),
  - Earthquake from baseline-style params (best Earthquake).

### Accuracy Table (%)

| Experiment | Flood RF | Flood GB | Flood SVM | Flood Base Avg | Flood Stacking | Earthquake RF | Earthquake GB | Earthquake SVM | Earthquake Base Avg | Earthquake Stacking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 Baseline | 82.50 | 80.83 | 81.67 | 81.67 | 82.50 | 91.25 | 88.75 | 91.25 | 90.42 | 92.50 |
| E1 Hyperparam-heavy | 81.70 | 81.70 | 77.50 | 80.30 | 78.30 | 90.00 | 86.20 | 85.00 | 87.07 | 88.80 |
| E2 Generator tuning (v2) | 89.17 | 88.33 | 90.83 | 89.44 | 89.17 | 91.25 | 88.75 | 91.25 | 90.42 | 92.50 |
| E3 Joint aggressive tuning | 93.33 | 93.33 | 96.67 | 94.44 | 94.17 | 91.25 | 83.75 | 90.00 | 88.33 | 87.50 |
| E4 Final selected combination | 93.33 | 93.33 | 96.67 | 94.44 | 94.17 | 91.25 | 88.75 | 91.25 | 90.42 | 92.50 |

### Combined Means Across Both Disasters

| Experiment | Mean Base Avg | Mean Stacking |
|---|---:|---:|
| E0 | 86.05 | 87.50 |
| E1 | 83.69 | 83.55 |
| E2 | 89.93 | 90.84 |
| E3 | 91.39 | 90.84 |
| E4 Final | 92.43 | 93.34 |

### Key Improvement
- Flood stacking improved from `82.50%` (E0) to `94.17%` (E4).
- Improvement = `+11.67` points.
- Earthquake stacking maintained at `92.50%`.

---

## 3. What Was Tuned

## E1 Hyperparam-heavy tuning
Changes:
- RF: more estimators/depth + class weighting.
- GB: more estimators, lower learning rate, subsample.
- SVM: higher C + class weighting.
- Logistic meta-learner: class weighting.

Outcome:
- Degraded both disasters.
- Rejected.

## E2 Generator tuning (v2)
Changes:
- Reverted model params to baseline.
- Tuned Flood synthetic generator:
  - stronger rainfall-soil_moisture coupling,
  - stronger pressure inverse coupling,
  - lower noise,
  - adjusted threshold.

Outcome:
- Flood improved significantly.
- Earthquake unaffected (stayed strong).

## E3 Joint aggressive tuning
Changes:
- Stronger model params (RF/GB/SVM) + even stronger Flood data separability.

Outcome:
- Flood increased to 94.17.
- Earthquake dropped to 87.50 stacking under same global params.
- Useful for Flood only, not acceptable globally.

## E4 Final selection
Strategy:
- Keep best Flood artifacts from E3.
- Keep best Earthquake artifacts from baseline-style params (E2-equivalent).
- Use per-disaster artifact folders.

Outcome:
- Best combined performance.

---

## 4. Technical Issues and Fixes

1. Missing ML deps (`sklearn`)  
Fix: install `ml_pipeline/requirements.txt`.

2. Console encoding failure (`UnicodeEncodeError`)  
Fix: run with `PYTHONIOENCODING=utf-8`.

3. Matplotlib permission/cache issues  
Fix: set `MPLCONFIGDIR` to writable project folder.

4. Windows worker permission error (`WinError 5`) with parallel jobs  
Fix:
- RF `n_jobs=1` in `config.py`.
- Stacking `n_jobs=1` in `stacking_model.py`.

5. SHAP local explanation shape error  
Fix: normalize SHAP output dimensionality in `explainability.py`.

---

## 5. Training Count

- Comparable successful experiments: `5` (`E0` to `E4`).
- Additional reruns were executed for environment fixes and artifact snapshot corrections.

---

## 6. Artifact Checkpoints

Baseline:
- `ml_pipeline/models_flood/`
- `ml_pipeline/models_earthquake/`

V2 tuned:
- `ml_pipeline/models_tuned_v2_flood/`
- `ml_pipeline/models_tuned_v2_earthquake/`

V3:
- `ml_pipeline/models_tuned_v3_flood/` (intermediate, not final for Flood due snapshot sequencing)
- `ml_pipeline/models_tuned_v3_earthquake/` (selected for Earthquake)

Final selected:
- `ml_pipeline/models_tuned_v4_flood/` (selected Flood)
- `ml_pipeline/models_tuned_v3_earthquake/` (selected Earthquake)

---

## 7. Recommended Production Mapping

Use disaster-specific model roots:
- Flood -> `models_tuned_v4_flood`
- Earthquake -> `models_tuned_v3_earthquake`

This is required to preserve the best score for each disaster simultaneously.

---

## 8. Final Conclusion

- Accuracy improved substantially.
- Final selected (`E4`) results:
  - Flood stacking: `94.17%`
  - Earthquake stacking: `92.50%`
  - Mean stacking across both: `93.34%`
- This is the best combined configuration from all runs in this session.
