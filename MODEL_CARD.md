# Model card: DiaTwin glucose forecaster and alerts

## Model details
- **Developers:** Team DiaTwin, IMT Ghaziabad. Version 1.0 (Digital Twin Challenge 2026 prototype). License: MIT.
- **Type:** gradient-boosted decision trees (scikit-learn `HistGradientBoosting`), seven models: 30/60/120-minute delta regressors, 10% and 90% quantile regressors (2 h), spike-onset classifier, hypoglycaemia classifier. Plus a mechanistic what-if layer and a feature-group explanation layer.
- **Inputs:** 43 features fusing static EHR fields with CGM, wearable and meal-diary streams (see `models/meta.json`).
- **Outputs:** glucose at +30/+60/+120 min, 80% band at 2 h, spike-onset risk, hypoglycaemia risk score, feature-group explanation.

## Intended use
Research and demonstration of a digital-twin concept for Type 2 Diabetes: decision support shown to a clinician on synthetic patients. **Not for diagnosis, dosing or any clinical decision.**

## Out-of-scope uses
Real patients; Type 1 diabetes, gestational or paediatric diabetes; automated insulin delivery; unsupervised alerts to patients.

## Training and evaluation data
Fully synthetic: 400 patients x 14 days at 5-minute resolution (1,612,800 readings), see the [Data card](DATA_CARD.md). Patient-level split 239 / 80 / 81, stratified by medication class; an independent shifted cohort of 120 patients was used for the distribution-shift test.

## Performance (test patients)
| Task | Result |
|---|---|
| 2-hour forecast MAE | 12.9 mg/dL (CI [12.4, 13.5]); persistence 26.5 |
| 30 / 60-minute MAE | 5.7 / 10.0 mg/dL |
| Cold-start 2-hour MAE | 13.1 mg/dL (CGM only 16.4) |
| Spike-onset ROC-AUC / PR-AUC | 0.929 / 0.797; ECE 0.009 |
| Spike alert at tuned threshold | recall 82%, precision 65%; 3.7 episodes/patient-day |
| Hypoglycaemia ROC-AUC | 0.970 (CGM only 0.973; rare event, 7 positive patients) |
| 5-fold CV MAE | 13.43 +/- 0.06 (CGM only 14.60 +/- 0.14) |

## Factors and subgroup analysis
Performance by HbA1c band, medication, age, sex and activity is in the [Technical report](TECHNICAL_REPORT.md) (Section 7.10). Among subgroups with at least 10 patients, fusion MAE ranges from 12.1 mg/dL (HbA1c: <7%) to 14.2 mg/dL (HbA1c: >=8%); every such subgroup improves on persistence by at least 46%. Groups with fewer than 10 patients are flagged and should not be over-read.

## Robustness
Tested without retraining under CGM noise, 20% CGM dropout, no meal diary and wearables offline. Modality-dropout training is essential: with wearables offline MAE is 13.1 mg/dL against 16.6 for plain training.

## Ethical considerations and risks
- Alert fatigue (see operating points), false reassurance when a stream silently fails (mitigated by data-quality warnings), and over-trust in forecasts made on synthetic data.
- The model has never seen real patients; subgroup fairness on real populations is unknown.
- The hypoglycaemia output is a ranking score, not a calibrated probability.

## Caveats and recommendations
Treat all accuracy figures as properties of the simulator. Before any clinical use: validate on real anonymised data, recalibrate thresholds with clinicians, monitor drift, and complete a regulatory review.
