# DiaTwin technical report

Digital Twin Challenge 2026, Phase 1. This report documents the clinical rationale, data, methods, evaluation and limits of the DiaTwin prototype. All numbers are generated from `results/metrics.json` (one run of `python -m src.train`).

## 1. Executive summary

DiaTwin is a per-patient digital twin for Type 2 Diabetes. It fuses a static EHR with dynamic CGM, wearable and meal-diary streams to (i) forecast glucose 30, 60 and 120 minutes ahead with an uncertainty band, (ii) warn of hyperglycaemic excursions and hypoglycaemia, (iii) explain each forecast, and (iv) simulate what-if choices. On 81 held-out synthetic patients the 2-hour forecast error is **12.9 mg/dL** versus 26.5 for persistence (-51%). Fusing the EHR and wearables improves on CGM alone by 1.22 mg/dL (statistically significant) and by **20% at cold start**. The model remains accurate when inputs degrade, provided it is trained with modality dropout. All results are on synthetic data and are not clinical claims.

## 2. Clinical background and rationale

- **Burden.** ICMR-INDIAB estimates 101 million Indians with diabetes and 136 million with prediabetes (Anjana et al., *Lancet Diabetes Endocrinol* 2023).
- **Why a two-hour horizon.** Post-meal glucose excursions peak roughly 45-90 minutes after eating and resolve within about three hours, so a two-hour warning leaves time for a clinically meaningful action (a walk, a smaller portion, a dose adjustment).
- **Why glucose targets are expressed as time in range.** International consensus on CGM metrics (Battelino et al., *Diabetes Care* 2019) defines time in range as 70-180 mg/dL, with time-below-range and time-above-range reported separately. DiaTwin uses the same thresholds (70 and 180 mg/dL) for its two adverse events.
- **Why a digital twin rather than a classifier.** A twin keeps a persistent, personalised state, supports counterfactual questions and degrades visibly when inputs break; a static risk score does none of these.

## 3. Problem formulation

Let g(t) be the physiological glucose and g_obs(t) what the CGM reports (with gaps). At each 5-minute step the twin sees the history of g_obs, the wearable and meal-diary streams, and the static EHR vector x. It estimates:

1. **Forecast:** g(t+h) - g_obs(t) for h in {30, 60, 120} min (the model predicts the change; the current reading is added back).
2. **Spike onset:** among rows with g_obs(t) <= 180, whether max g over (t, t+120 min] exceeds 180 mg/dL. Restricting to rows that are not already high avoids rewarding trivial persistence.
3. **Hypoglycaemia onset:** among rows with g_obs(t) >= 70, whether min g over (t, t+120 min] falls below 70 mg/dL.

Targets use the noise-free physiological glucose, so sensor noise and dropouts are a property of the inputs only.

## 4. Data

### 4.1 Synthetic EHR

400 patients; age ~ N(52, 11), BMI ~ N(27.5, 4.2), disease duration ~ Gamma(2.2, 3.0) years, TCF7L2 risk alleles in {0, 1, 2} with probabilities (0.45, 0.43, 0.12). HbA1c = 6.5 + 0.045 (BMI - 25) + 0.06 duration + 0.20 alleles + N(0, 0.7). Fasting glucose = 0.82 x eAG + noise, with eAG = 28.7 HbA1c - 46.7 (the ADA conversion). eGFR, LDL, hypertension, family history, activity level, diet and sleep habit complete the record. Medication escalates with HbA1c (none / metformin, then + sulfonylurea, then insulin). Mix: metformin 60%, metformin+sulfonylurea 24%, none 10%, insulin 6%.

### 4.2 Physiology simulator

Per patient, at 5-minute resolution:

```
glucose(t) = basal(t) + sum_meals R(t - t_m; carbs_m, ISF, GI) - sum_doses K(t - t_d; planned_carbs, ISF) - E(steps) + AR(1) noise + sensor noise
basal(t)   = fasting + 12 exp(-(hour - 6.5)^2 / (2 x 1.3^2)) + 5 max(0, 7 - sleep_prev) + 9 stress          (dawn effect, sleep debt, stress)
R(tau)     = 1.25 x carbs x GI / ISF x (tau/55)^2 exp(2 (1 - tau/55))                                       (peak at 55 min)
K(tau)     = k x R with peak 45 min (insulin, k = 1.35) or 100 min (sulfonylurea, k = 0.55), dosed for the PLANNED meal
E(steps)   = 0.013 x sum_k steps(t-k) exp(-k/10), k = 0..17                                                  (about 90 min of effect)
ISF        = exp(-0.18 (HbA1c-7.5) - 0.035 (BMI-27) - 0.008 (age-52) - 0.02 (duration-5) - 0.10 alleles) x f(medication) x f(activity)
```

The drug term is what creates hypoglycaemia: a dose is taken for a planned meal that may be skipped (about 4% of meals plus the planned slot) or eaten smaller, and doses are sometimes taken for skipped meals (50%). Heart rate and HRV follow activity, sleep, stress, age and BMI. **Imperfections** are modelled deliberately: 85% of meals are logged in the diary, and CGM dropouts remove short segments (about 1.6% of readings).

### 4.3 Realism check

| Metric | DiaTwin cohort | Consensus reference |
|---|---|---|
| Time in range 70-180 mg/dL | 86% | > 70% (target for most adults) |
| Time above 180 mg/dL | 14% | < 25% |
| Time above 250 mg/dL | 1.3% | < 5% |
| Time below 70 mg/dL | 0.12% | < 4% |
| Time below 54 mg/dL | 0.07% | < 1% |
| Mean glucose | 149 mg/dL | - |
| Glucose management indicator (GMI) | 6.9% | - |
| Coefficient of variation | 17% | <= 36% (stable) |
| CGM dropout | 1.6% of readings | - |

The cohort sits comfortably in the consensus-recommended range, which means it is **better controlled and less variable** than typical real-world Indian T2D (population mean HbA1c is higher in practice, and real CGM CV is usually above the 17% simulated here). Absolute errors will therefore be optimistic. The distribution-shift cohort (Section 7.9) partly addresses this by testing an older, heavier, longer-duration population.

### 4.4 Synthea compatibility

`src/import_synthea.py` maps Synthea's `patients.csv`, `conditions.csv` and `observations.csv` (LOINC 4548-4 HbA1c, 39156-5 BMI, 2339-0 glucose, 33914-3 eGFR, 18262-6 LDL) onto the same schema, then fills fields Synthea does not model (genetic marker, lifestyle) from the generator's distributions. It is unit-tested on mock CSVs in Synthea's layout and the output is verified to flow through the simulator and feature pipeline.

## 5. Features (43 fused features)

| Group | Count | Examples |
|---|---|---|
| CGM history and trend | 17 | current glucose, lags (5-120 min), 15/30/60-min deltas, 1 h and 3 h rolling stats, 24 h mean, CGM gap, hour of day (sin, cos) |
| Meal diary | 3 | logged carbs in last 1 h / 3 h, minutes since last meal |
| Wearables | 7 | steps (15 min, 1 h, 3 h), heart rate, HRV, sleep hours (24 h), HRV during sleep |
| EHR profile | 16 | age, sex, BMI, duration, HbA1c, fasting glucose, eGFR, LDL, hypertension, family history, TCF7L2 alleles, medication flags, activity, diet |

Short CGM gaps (up to 30 minutes) are bridged by carrying the last reading forward; longer gaps leave the value missing and the model handles it natively. Missing wearables stay missing (not zero), so a switched-off device is distinguishable from a sedentary one.

## 6. Models

- **Learned layer.** `HistGradientBoostingRegressor` / `Classifier` (300 iterations max, learning rate 0.06, 31 leaves, L2 = 1.0, early stopping on 10% internal validation). Native handling of missing values makes it a natural fit for streams that drop out.
- **Delta target.** Predicting g(t+h) - g_obs(t) rather than g(t+h) removes the dominant persistence component and lets the trees spend capacity on the dynamics.
- **Uncertainty.** Two quantile regressors (10%, 90%) give an 80% band; empirical coverage is 79%.
- **Classifiers.** Spike onset uses the tuned threshold (0.33, maximising F1 on validation patients). Hypoglycaemia uses `class_weight='balanced'` and a recall-oriented threshold (0.02, maximising F2), so its output is a ranking score rather than a calibrated probability.
- **Modality-dropout training.** In training rows, wearables are masked in 15% of rows, the meal diary is blanked in 15% and both in 5%. This is the difference between a model that collapses when a stream disappears and one that falls back gracefully (Section 7.8).
- **Mechanistic layer.** Reuses the simulator's kernels with the patient's insulin-sensitivity index to produce counterfactual 3-hour curves: eat, eat half, eat then walk, skip. It starts from the current reading and adds the residual response of meals already logged.
- **Explanation layer.** For each feature group the group is replaced by its training median and the change in the predicted 2-hour delta and spike risk is reported (positive = pushes glucose up).

## 7. Evaluation and results

**Protocol.** 60/20/20 split by patient, stratified by medication class so that rare insulin users appear in every split (239 / 80 / 81 patients). The first 24 h of each series is history only. Training and test rows are sampled every 15 minutes. All confidence intervals are patient-level bootstrap (whole patients resampled).

### 7.1 Ablation: forecast

| Model | MAE (mg/dL) | 95% CI | RMSE | within 15 mg/dL | within 20% |
|---|---|---|---|---|---|
| Persistence baseline | 26.5 | [25.1, 27.9] | 38.4 | 49% | 67% |
| Linear-trend extrapolation | 33.9 | - | 50.7 | 41% | 63% |
| CGM only | 14.2 | [13.6, 14.8] | 21.5 | 69% | 89% |
| CGM + wearables | 13.9 | - | 21.4 | 70% | 89% |
| CGM + EHR | 13.3 | - | 20.6 | 72% | 90% |
| **Full fusion (final model)** | 12.9 | [12.4, 13.5] | 20.2 | 73% | 90% |

Fusion vs CGM-only: paired MAE gain 1.22 mg/dL, 95% CI [1.0, 1.4] (statistically significant).

### 7.2 Ablation: spike-onset alert

Prevalence in the test set 23%.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CGM only | 0.914 | 0.744 | 61% | 81% | 0.70 |
| CGM + wearables | 0.921 | 0.771 | 65% | 79% | 0.71 |
| CGM + EHR | 0.922 | 0.769 | 64% | 81% | 0.71 |
| **Full fusion (final model)** | 0.929 | 0.797 | 65% | 82% | 0.72 |

AUC gain of fusion over CGM-only 0.014 (CI [0.010, 0.020]), statistically significant. Calibration of the final spike classifier: Brier 0.089, ECE 0.009.

![calibration](../results/figures/calibration_spike.png)

### 7.3 Cold start

| Model | MAE (mg/dL) | RMSE | within 20% |
|---|---|---|---|
| CGM only (< 1 h) | 16.4 | 24.2 | 84% |
| CGM + wearables | 15.4 | 23.2 | 87% |
| CGM + EHR | 13.6 | 20.9 | 89% |
| **Full fusion** | 13.1 | 20.4 | 90% |

### 7.4 Model families

| Model family (same fusion features) | MAE (mg/dL) | RMSE |
|---|---|---|
| HistGradientBoosting (ours) | 12.9 | 20.2 |
| Ridge regression | 17.4 | 24.3 |
| Extra-trees (80) | 13.2 | 20.4 |
| MLP (64-32) | 13.4 | 20.8 |

### 7.5 Multi-horizon

| Horizon | Persistence | CGM only | Full fusion | Fusion vs persistence |
|---|---|---|---|---|
| 30 min | 11.1 | 7.2 | **5.7** | -48% |
| 60 min | 18.6 | 11.5 | **10.0** | -46% |
| 120 min | 26.5 | 14.2 | **12.9** | -51% |

![horizon](../results/figures/horizon_mae.png)

### 7.6 Alert burden, lead time and operating points

Alert episodes are runs of consecutive alerting rows; an episode is *false* if glucose does not cross 180 mg/dL within the following 2 hours; an excursion is *flagged* if an alert fired in the preceding 2 hours, and its lead time is the time from the first such alert to the crossing (capped at 120 minutes by the look-back).

| Threshold | Alert episodes / patient-day | False-alert share | Excursions flagged | Flagged >= 30 min ahead | Median lead (min) | Row precision | Row recall |
|---|---|---|---|---|---|---|---|
| 0.20 | 3.9 | 70% | 99.9% | 98% | 120 | 57% | 90% |
| 0.33 (tuned) | 3.7 | 58% | 99.5% | 96% | 120 | 65% | 82% |
| 0.50 | 3.5 | 43% | 97.9% | 91% | 105 | 73% | 69% |
| 0.70 | 3.1 | 26% | 92.9% | 79% | 75 | 83% | 49% |
| 0.85 | 2.3 | 10% | 79.8% | 51% | 30 | 92% | 22% |

![lead](../results/figures/alert_lead_time.png)

### 7.7 Hypoglycaemia

312 positive rows from 7 test patients (6 on insulin). Fusion ROC-AUC 0.970 (CI [0.936, 0.997]) vs CGM-only 0.973; fusion recall 89% at precision 9%, PR-AUC 0.17 against a base rate of 0.31%. The two AUCs are statistically indistinguishable (confidence intervals overlap): CGM history alone already reveals a falling trend, so the EHR adds no discrimination here. Fusion's benefit appears at the chosen operating point (recall 89% vs 60% for CGM only, both at about 9% precision). The recall-oriented threshold sits at the lowest value in the search grid, so the alert amounts to flagging any non-trivial risk; The low precision and the small number of positive patients mean this is a proof of concept.

### 7.8 Robustness

Scenarios degrade the observed streams of the test patients; the targets are unchanged and models are not retrained.

| Scenario | Persistence | CGM only | Fusion (plain training) | **Fusion (modality-dropout, final)** | Spike AUC (final) |
|---|---|---|---|---|---|
| Clean | 26.5 | 14.2 | 13.0 | **12.9** | 0.929 |
| CGM noise (+8 mg/dL SD) | 27.7 | 16.3 | 14.8 | **14.8** | 0.904 |
| 20% CGM dropout | 26.6 | 14.4 | 13.1 | **13.1** | 0.927 |
| Meal diary not used | 26.5 | 14.2 | 13.8 | **13.4** | 0.921 |
| Wearables offline | 26.5 | 14.2 | 16.6 | **13.1** | 0.929 |

Plain fusion training fails when wearables disappear (16.6 mg/dL, worse than CGM only at 14.2); modality-dropout training restores it (13.1). CGM noise raises MAE from 12.9 to 14.8 mg/dL for the final fusion model and from 14.2 to 16.3 for CGM only; 20% CGM dropout and an unused meal diary cost little (13.1 and 13.4).

![robust](../results/figures/robustness_mae.png)

### 7.9 Distribution shift

Unseen cohort (120 patients; age 63 vs 52, BMI 30.1 vs 27.7, duration 9.9 vs 6.6 y, HbA1c 7.5 vs 7.2): persistence 33.7, CGM only 18.1, fusion 16.4 mg/dL, spike AUC 0.926. The ranking is preserved under shift; absolute error rises because this population is intrinsically harder to forecast.

### 7.10 Subgroups

| Dimension | Group | Patients | MAE persistence | MAE CGM only | MAE fusion | Spike ROC-AUC |
|---|---|---|---|---|---|---|
| HbA1c | 7-8% | 31 | 28.4 | 14.3 | **13.3** | 0.928 |
| HbA1c | <7% | 35 | 25.0 | 12.9 | **12.1** | 0.935 |
| HbA1c | >=8% | 15 | 26.1 | 16.8 | **14.2** | 0.906 |
| Medication | insulin (small n) | 6 | 21.3 | 16.5 | **13.4** | 0.874 |
| Medication | metformin / none | 56 | 27.9 | 13.8 | **13.1** | 0.932 |
| Medication | sulfonylurea | 19 | 23.9 | 14.6 | **12.4** | 0.934 |
| Age | 50-64 | 32 | 25.2 | 13.9 | **12.6** | 0.927 |
| Age | <50 | 39 | 26.4 | 14.3 | **13.0** | 0.930 |
| Age | >=65 | 10 | 30.6 | 14.4 | **13.8** | 0.937 |
| Sex | female | 41 | 27.5 | 14.5 | **13.3** | 0.925 |
| Sex | male | 40 | 25.5 | 13.9 | **12.6** | 0.933 |
| Activity | light | 32 | 25.1 | 13.9 | **12.6** | 0.933 |
| Activity | moderate | 17 | 25.5 | 14.1 | **12.8** | 0.924 |
| Activity | sedentary | 32 | 28.3 | 14.5 | **13.3** | 0.929 |

Among subgroups with at least 10 patients, fusion MAE ranges from 12.1 mg/dL (HbA1c: <7%) to 14.2 mg/dL (HbA1c: >=8%); every such subgroup improves on persistence by at least 46%. Groups with fewer than 10 patients are flagged and should not be over-read.

![sub](../results/figures/subgroup_mae.png)

### 7.11 Cross-validation

| Model | Mean MAE | SD across folds |
|---|---|---|
| persistence | 27.32 | 0.27 |
| cgm_only | 14.60 | 0.14 |
| fusion | 13.43 | 0.06 |
| cold_cgm | 16.98 | 0.10 |
| cold_fusion | 13.54 | 0.07 |

### 7.12 Importance

Top features by permutation importance: `g_now`, `g_lag5`, `hour_sin`, `hour_cos`, `fasting_glucose`, `g_mean24h`, `g_std1h`, `med_sulfonylurea`. By group:

- **Glucose history & trend (CGM)**: +22.10 mg/dL MAE when shuffled
- **EHR profile**: +4.21 mg/dL MAE when shuffled
- **Meals (diary)**: +1.32 mg/dL MAE when shuffled
- **Activity, HR, HRV, sleep (wearables)**: +0.48 mg/dL MAE when shuffled

![imp](../results/figures/feature_importance.png)

## 8. Safety, ethics and regulation

- **Human in the loop.** DiaTwin is decision support for a clinician. It never doses, and it makes its uncertainty visible (80% band, risk tiers, data-quality warnings).
- **Alert fatigue.** The default threshold is sensitive and noisy; Section 7.6 exposes the trade-off so clinicians can choose an operating point. A deployed system should add snooze logic and per-patient thresholds.
- **Fairness.** Performance is reported by HbA1c band, medication, age, sex and activity (Section 7.10). Real-world fairness analysis needs real demographics, including region, language and device access.
- **Failure behaviour.** CGM gaps, offline wearables and unlogged meals are detected and reported, and the model was trained to cope with them. The dashboard's data-quality simulator demonstrates this.
- **Privacy.** No real patient data is used; DPDP Act and HIPAA constraints are satisfied by construction. A real deployment would require consent, data minimisation and on-device or in-region processing.
- **Regulation.** A tool that influences treatment decisions would likely be regulated as software as a medical device; the applicable pathway must be assessed with qualified advisers before any clinical use.

## 9. Limitations and threats to validity

1. **Synthetic data.** Everything shares the simulator's assumptions. The simulated CGM variability (CV 17%) and control (TIR 86%) are more benign than typical real-world T2D, so real errors will be larger.
2. **Shared assumptions in the what-if layer.** The mechanistic layer reuses the simulator's kernels; its fidelity on real patients is unvalidated.
3. **Wearable value is understated or overstated.** In the simulator wearables act mainly through steps; in reality they may carry more or less independent information.
4. **Hypoglycaemia.** Few events from few insulin users; the alert threshold is tuned for recall at the cost of precision.
5. **Meal logging.** The twin cannot anticipate unlogged meals, the largest source of residual error.
6. **Single simulator family.** The distribution-shift test changes the cohort, not the generative process.

## 10. Roadmap

1. Validate on real anonymised CGM and EHR data (open CGM datasets, MIMIC-IV for EHR context) and recalibrate.
2. Learn the mechanistic layer's parameters from data instead of fixing them.
3. Add medication-timing and insulin-dose inputs; sequence models as an additional model family.
4. Clinician usability study on the dashboard; per-patient alert thresholds; drift monitoring.
5. Regulatory and information-governance review.

## 11. Reproducibility

`python -m src.train` regenerates every artefact from seeds (EHR seed 42, simulator seed 7, split seed 0). `pytest -q` runs 14 tests, including physiology monotonicity, target-leakage checks, the Synthea importer, alert-metric arithmetic and graceful degradation. CI runs the tests and a dashboard smoke test on every push.

## 12. References

1. Anjana RM et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB cross-sectional study. *Lancet Diabetes & Endocrinology*, 2023.
2. Battelino T et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. *Diabetes Care*, 2019.
3. Bergenstal RM et al. Glucose management indicator (GMI). *Diabetes Care*, 2018.
4. Walonoski J et al. Synthea: an approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. *JAMIA*, 2018.
5. Johnson AEW et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 2023.
6. Pedregosa F et al. Scikit-learn: machine learning in Python. *JMLR*, 2011.
