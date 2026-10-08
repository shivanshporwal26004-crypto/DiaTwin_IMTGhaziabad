# Data card: DiaTwin synthetic cohort

## Overview
Fully synthetic, generated on demand by `python -m src.train` from fixed seeds. Contains **no real patient data and no personal identifiers**. Committed files: `data/ehr.csv` (all patients) and `data/demo_*` (12 held-out demo patients).

## Composition
| | |
|---|---|
| Patients | 400 (+ 120 in a shifted evaluation cohort) |
| Duration | 14 days per patient |
| Resolution | 5 minutes |
| Sensor rows | 1,612,800 |
| Medication mix | metformin 60%, metformin+sulfonylurea 24%, none 10%, insulin 6% |

### EHR table (`data/ehr.csv`)
`patient_id, age, sex, bmi, duration_years, hba1c, fasting_glucose, egfr, ldl, hypertension, family_history, tcf7l2_risk_alleles, medication, activity_level, vegetarian, carb_scale, sleep_hours_mean`

### Sensor table (`data/demo_sensors.csv.gz`)
| Column | Meaning |
|---|---|
| `ts` | timestamp (5-minute grid) |
| `glucose` | CGM reading as observed; NaN during sensor dropouts |
| `glucose_true` | noise-free physiological glucose (used only for targets and the "reveal" view) |
| `hr`, `hrv` | heart rate (bpm), HRV RMSSD (ms) |
| `steps` | steps per 5 minutes |
| `sleep_stage` | 0 awake, 1 light, 2 deep, 3 REM |
| `meal_carbs_g` | carbohydrates logged in the diary at the meal time (NaN if unlogged; about 85% of meals are logged) |

## Generation process
See the [Technical report](TECHNICAL_REPORT.md) (Section 4) for the full equations. In short: EHR variables are drawn from clinically plausible distributions with HbA1c-dependent medication; a physiology simulator turns each patient's insulin sensitivity into meal, drug and exercise responses with dawn effect, sleep debt, stress, AR(1) noise, CGM noise and dropouts.

## Glycaemic realism
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

## Known biases and limitations
- **Too benign.** TIR 86% and CV 17% are better and tighter than typical real-world T2D, especially in India where mean HbA1c is higher.
- **Simplified physiology.** One meal kernel shape, no fat/protein effects, no illness, no alcohol, no insulin-dose timing errors beyond a mismatch factor.
- **Regular routines.** Meal times are tightly scheduled, so time of day is a strong predictor.
- **Few insulin users** (6% of the cohort), so hypoglycaemia events are rare and concentrated.
- **No demographic diversity beyond the modelled variables** (no region, language, device type).

## Intended use
Prototyping and evaluating digital-twin methods under the Digital Twin Challenge sandbox rules. Not for estimating real-world clinical performance.

## Privacy and compliance
No real data, so DPDP Act and HIPAA restrictions are met by construction.

## Maintenance and license
Regenerate with `python -m src.train`. MIT License.
