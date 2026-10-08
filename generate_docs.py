"""Generates README.md, docs/TECHNICAL_REPORT.md, docs/MODEL_CARD.md and docs/DATA_CARD.md from
results/metrics.json so that every number in the documentation comes from the same run.

    python docs/generate_docs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "metrics.json").read_text())
META = json.loads((ROOT / "models" / "meta.json").read_text())

D, R, C, HY = M["dataset"], M["regression"], M["classification"], M["hypoglycaemia"]
CS, FAM, HZ, BO = M["cold_start"], M["model_families"], M["horizons"], M["bootstrap"]
AL, OP, SG, RB = M["alerts"], M["operating_points"], M["subgroups"], M["robustness"]
SH, CV, GI, TF, GL = M["distribution_shift"], M["cross_validation"], M["group_importance"], M["top_features"], D["glycemic"]


def f(v, d=1):
    return f"{v:.{d}f}"


def pct_cut(a, b):
    return round((1 - b / a) * 100)


gain = BO["mae_fusion_vs_cgm"]["fusion"]["gain_vs_cgm_only"]
auc_gain = BO["spike_auc_gain_fusion_vs_cgm"]
sig_mae = "statistically significant" if gain["ci95"][0] > 0 else "not statistically significant"
sig_auc = "statistically significant" if auc_gain["ci95"][0] > 0 else "not statistically significant"
cut_persist = pct_cut(R["persistence"]["mae"], R["fusion"]["mae"])
cut_cold = pct_cut(CS["cold_cgm"]["mae"], CS["cold_fusion"]["mae"])
tuned = next(o for o in OP if o["tuned"])
rb_w = RB["wearables_offline"]
best_fam = min(FAM, key=lambda k: FAM[k]["mae"])
a_f = AL["fusion"]
cvf, cvc = CV["fusion"], CV["cgm_only"]
dsn = D["train_val_test_patients"]
lab = {"cgm_only": "CGM only", "cgm+wearables": "CGM + wearables", "cgm+ehr": "CGM + EHR", "fusion": "**Full fusion (final model)**"}
hy_f, hy_c = HY["fusion"]["roc_auc"], HY["cgm_only"]["roc_auc"]
hypo_helps = hy_f - hy_c > 0.01
HYPO_TXT = (f"The EHR (medication) improves hypoglycaemia discrimination (AUC {f(hy_f, 3)} vs {f(hy_c, 3)})." if hypo_helps else
            f"The two AUCs are statistically indistinguishable (confidence intervals overlap): CGM history alone already reveals a falling trend, so the EHR adds no discrimination here. "
            f"Fusion's benefit appears at the chosen operating point (recall {f(HY['fusion']['recall'] * 100, 0)}% vs {f(HY['cgm_only']['recall'] * 100, 0)}% for CGM only, both at about {f(HY['fusion']['precision'] * 100, 0)}% precision).")
HYPO_THR = ("The recall-oriented threshold sits at the lowest value in the search grid, so the alert amounts to flagging any non-trivial risk; " if META["hypo_threshold_prob"] <= 0.021 else "") 
_big = [r for r in SG if r["patients"] >= 10]
_w, _b = max(_big, key=lambda r: r["mae_fusion"]), min(_big, key=lambda r: r["mae_fusion"])
_imp = min(pct_cut(r["mae_persistence"], r["mae_fusion"]) for r in _big)
SUBTXT = (f"Among subgroups with at least 10 patients, fusion MAE ranges from {f(_b['mae_fusion'])} mg/dL ({_b['dimension']}: {_b['group']}) to {f(_w['mae_fusion'])} mg/dL "
          f"({_w['dimension']}: {_w['group']}); every such subgroup improves on persistence by at least {_imp}%. Groups with fewer than 10 patients are flagged and should not be over-read.")
_n = RB["cgm_noise_8mgdl"]
RBN = (f"CGM noise raises MAE from {f(RB['clean']['fusion_mae'])} to {f(_n['fusion_mae'])} mg/dL for the final fusion model and from {f(RB['clean']['cgm_only_mae'])} to {f(_n['cgm_only_mae'])} for CGM only; "
       f"20% CGM dropout and an unused meal diary cost little ({f(RB['cgm_dropout_20pct']['fusion_mae'])} and {f(RB['no_meal_log']['fusion_mae'])}).")
NTESTS = (ROOT / "tests" / "test_pipeline.py").read_text().count("\ndef test_")
ci = lambda d: f"[{f(d['ci95'][0])}, {f(d['ci95'][1])}]"
ci3 = lambda d: f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"

# ------------------------------------------------------------------ shared tables
abl_rows = []
for k, name in [("persistence", "Persistence baseline"), ("linear_trend", "Linear-trend extrapolation"), ("cgm_only", lab["cgm_only"]),
                ("cgm+wearables", lab["cgm+wearables"]), ("cgm+ehr", lab["cgm+ehr"]), ("fusion", lab["fusion"])]:
    cik = ci(BO["mae"][k]) if k in BO["mae"] else "-"
    abl_rows.append(f"| {name} | {f(R[k]['mae'])} | {cik} | {f(R[k]['rmse'])} | {f(R[k]['within15mgdl_pct'], 0)}% | {f(R[k]['within20_pct'], 0)}% |")
ABL = "| Model | MAE (mg/dL) | 95% CI | RMSE | within 15 mg/dL | within 20% |\n|---|---|---|---|---|---|\n" + "\n".join(abl_rows)

cls_rows = "\n".join(
    f"| {lab[k]} | {f(C[k]['roc_auc'], 3)} | {f(C[k]['pr_auc'], 3)} | {f(C[k]['precision'] * 100, 0)}% | {f(C[k]['recall'] * 100, 0)}% | {f(C[k]['f1'], 2)} |"
    for k in ["cgm_only", "cgm+wearables", "cgm+ehr", "fusion"])
CLS = "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n" + cls_rows

cl = {"cold_cgm": "CGM only (< 1 h)", "cold_cgm+wearables": "CGM + wearables", "cold_cgm+ehr": "CGM + EHR", "cold_fusion": "**Full fusion**"}
COLD = "| Model | MAE (mg/dL) | RMSE | within 20% |\n|---|---|---|---|\n" + "\n".join(
    f"| {cl[k]} | {f(CS[k]['mae'])} | {f(CS[k]['rmse'])} | {f(CS[k]['within20_pct'], 0)}% |" for k in cl)

FAMT = "| Model family (same fusion features) | MAE (mg/dL) | RMSE |\n|---|---|---|\n" + "\n".join(
    f"| {k} | {f(v['mae'])} | {f(v['rmse'])} |" for k, v in FAM.items())

HZT = "| Horizon | Persistence | CGM only | Full fusion | Fusion vs persistence |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {h} min | {f(v['persistence']['mae'])} | {f(v['cgm_only']['mae'])} | **{f(v['fusion']['mae'])}** | -{pct_cut(v['persistence']['mae'], v['fusion']['mae'])}% |"
    for h, v in HZ.items())

OPT = "| Threshold | Alert episodes / patient-day | False-alert share | Excursions flagged | Flagged >= 30 min ahead | Median lead (min) | Row precision | Row recall |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(
    f"| {o['threshold']:.2f}{' (tuned)' if o['tuned'] else ''} | {f(o['alert_episodes_per_patient_day'])} | {f(o['false_alert_share_pct'], 0)}% | {f(o['detected_pct'], 1)}% | "
    f"{f(o['detected_with_30min_lead_pct'], 0)}% | {f(o['median_lead_min'], 0)} | {f(o['row_precision'] * 100, 0)}% | {f(o['row_recall'] * 100, 0)}% |" for o in OP)

SGT = "| Dimension | Group | Patients | MAE persistence | MAE CGM only | MAE fusion | Spike ROC-AUC |\n|---|---|---|---|---|---|---|\n" + "\n".join(
    f"| {r['dimension']} | {r['group']}{' (small n)' if r['patients'] < 10 else ''} | {r['patients']} | {f(r['mae_persistence'])} | {f(r['mae_cgm_only'])} | **{f(r['mae_fusion'])}** | "
    f"{f(r['spike_auc_fusion'], 3) if r['spike_auc_fusion'] else '-'} |" for r in SG)

nice = {"clean": "Clean", "cgm_noise_8mgdl": "CGM noise (+8 mg/dL SD)", "cgm_dropout_20pct": "20% CGM dropout", "no_meal_log": "Meal diary not used", "wearables_offline": "Wearables offline"}
RBT = "| Scenario | Persistence | CGM only | Fusion (plain training) | **Fusion (modality-dropout, final)** | Spike AUC (final) |\n|---|---|---|---|---|---|\n" + "\n".join(
    f"| {nice[k]} | {f(v['persistence_mae'])} | {f(v['cgm_only_mae'])} | {f(v['fusion_plain_mae'])} | **{f(v['fusion_mae'])}** | {f(v['fusion_spike_auc'], 3)} |" for k, v in RB.items())

CVT = "| Model | Mean MAE | SD across folds |\n|---|---|---|\n" + "\n".join(
    f"| {k} | {f(v['mean'], 2)} | {f(v['sd'], 2)} |" for k, v in CV.items())

GLT = f"""| Metric | DiaTwin cohort | Consensus reference |
|---|---|---|
| Time in range 70-180 mg/dL | {f(GL['tir_70_180_pct'], 0)}% | > 70% (target for most adults) |
| Time above 180 mg/dL | {f(GL['tar_over_180_pct'], 0)}% | < 25% |
| Time above 250 mg/dL | {f(GL['tar_over_250_pct'], 1)}% | < 5% |
| Time below 70 mg/dL | {f(GL['tbr_under_70_pct'], 2)}% | < 4% |
| Time below 54 mg/dL | {f(GL['tbr_under_54_pct'], 2)}% | < 1% |
| Mean glucose | {f(GL['mean_glucose_mgdl'], 0)} mg/dL | - |
| Glucose management indicator (GMI) | {f(GL['gmi_pct'], 1)}% | - |
| Coefficient of variation | {f(GL['cv_pct'], 0)}% | <= 36% (stable) |
| CGM dropout | {f(GL['cgm_dropout_pct'], 1)}% of readings | - |"""

TOPF = ", ".join(f"`{k}`" for k in list(TF)[:8])
GRP = "\n".join(f"- **{k}**: +{f(v, 2)} mg/dL MAE when shuffled" for k, v in GI.items())

TEAM = """| | |
|---|---|
| **Team name** | DiaTwin |
| **Team leader** | Shivansh Porwal |
| **Team members** | _(add names here)_ |
| **College** | IMT Ghaziabad (Institute of Management Technology), PGDM program |
| **Incubator** | Not applicable |"""

FIGS = "results/figures"

# ====================================================================== README
README = f"""# DiaTwin: A Digital Twin for Type 2 Diabetes

> Forecast glucose **30, 60 and 120 minutes ahead**, alert **before** a spike or a low, **explain why**, and let the doctor **simulate what-if choices**, by fusing a patient's EHR with wearable and CGM data.

Submission for the **Digital Twin Challenge 2026** (Happiest Health, on Unstop), Phase 1: Prototype & Code Submission.

![Architecture](docs/architecture_diagram.png)

## Highlights

- **Two fused data streams** (challenge requirement): static EHR + dynamic CGM / wearables / meal diary, {D['patients']} synthetic patients x {D['days_per_patient']} days = {D['sensor_rows']:,} readings.
- **Forecast error {f(R['fusion']['mae'])} mg/dL at 2 h** (95% CI {ci(BO['mae']['fusion'])}), **-{cut_persist}%** versus persistence; paired gain over CGM-only {f(gain['mean'], 2)} mg/dL (CI {ci(gain)}), {sig_mae}.
- **Cold start:** for a newly onboarded patient the EHR cuts error by **{cut_cold}%** ({f(CS['cold_cgm']['mae'])} to {f(CS['cold_fusion']['mae'])} mg/dL).
- **Spike-onset alert** ROC-AUC {f(C['fusion']['roc_auc'], 3)} (CI {ci3(BO['spike_auc']['fusion'])}), well calibrated (ECE {f(M['calibration']['spike']['ece'], 3)}); **hypoglycaemia alert** ROC-AUC {f(HY['fusion']['roc_auc'], 3)} on a rare event (CGM only {f(HY['cgm_only']['roc_auc'], 3)}).
- **Robust by design:** modality-dropout training keeps error at {f(rb_w['fusion_mae'])} mg/dL with wearables offline, where plain training degrades to {f(rb_w['fusion_plain_mae'])}.
- **Rigorous evaluation:** patient-level medication-stratified split, bootstrap CIs, 5 subgroup dimensions (HbA1c, medication, age, sex, activity), 4 robustness scenarios, a distribution-shift cohort, 5-fold CV, calibration, alert burden and lead time.
- **Explainable and honest:** per-patient "why" panel, operating-point trade-offs, and an explicit limitations section.

## Submission checklist (where each required item lives)

| Required item | Where |
|---|---|
| Team details | [1. Team details](#1-team-details) |
| College / incubator information | [1. Team details](#1-team-details) |
| Project title | [2. Project title](#2-project-title) |
| Problem statement | [3. Problem statement](#3-problem-statement) |
| Healthcare use case | [4. Healthcare use case](#4-healthcare-use-case) |
| Technical stack | [6. Technical stack](#6-technical-stack) |
| AI/ML model or framework details | [7. AI/ML model and framework details](#7-aiml-model-and-framework-details) |
| 15-20 minute demo video (unlisted YouTube) | [10. Demo video](#10-demo-video) |
| Open-source license details | [13. Open-source license](#13-open-source-license) (MIT, see [LICENSE](LICENSE)) |
| Architecture diagram (PDF/PPT) | [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf) (2 pages) |
| Presentation (PDF/PPT) | [`docs/DiaTwin_Presentation.pptx`](docs/DiaTwin_Presentation.pptx) and [`.pdf`](docs/DiaTwin_Presentation.pdf) |
| Files and links publicly accessible | This repository is public; every file above is in it, no permissions needed |

Extra documentation: [Technical report](docs/TECHNICAL_REPORT.md) | [Model card](docs/MODEL_CARD.md) | [Data card](docs/DATA_CARD.md) | [Demo script](docs/DEMO_SCRIPT.md)

## 1. Team details

{TEAM}

## 2. Project title

**DiaTwin: A Digital Twin for Type 2 Diabetes Glucose Forecasting, Early Warning and What-If Simulation**

## 3. Problem statement

About **101 million Indians live with diabetes and 136 million more have prediabetes** (ICMR-INDIAB study, *The Lancet Diabetes & Endocrinology*, 2023). Care is largely **reactive**: clinicians see periodic lab values such as HbA1c and occasional glucose readings, but not what the next two hours hold for a given patient. Post-meal spikes, and lows in patients on insulin or sulfonylureas, go unseen until harm accumulates.

## 4. Healthcare use case

**Condition:** Type 2 Diabetes (chronic, lifestyle-linked, highly prevalent in India), as the challenge requires one specific condition.

**Adverse events predicted**
- **Hyperglycaemic excursion:** glucose crossing **180 mg/dL** within 2 hours (upper bound of time-in-range).
- **Hypoglycaemia:** glucose dropping below **70 mg/dL** within 2 hours.
- Plus continuous **30 / 60 / 120-minute glucose forecasts** with an 80% uncertainty band.

**How it is used**
1. A patient is onboarded with their EHR; the twin personalises itself from it (day one, before much sensor history exists).
2. Wearable, CGM and meal-diary data stream in; the twin updates continuously and **flags missing streams** instead of silently guessing.
3. The doctor's dashboard shows forecasts, hyper/hypo risk, a **"why this forecast"** explanation and a ranked **cohort board** of who needs attention.
4. The doctor or patient asks "what if I eat this, walk after, or skip it?" and sees the next three hours under each choice.

## 5. The two data streams (challenge requirement)

| Stream | Content | Source in this prototype |
|---|---|---|
| **Static / historical** | demographics, BMI, disease duration, HbA1c, fasting glucose, eGFR, LDL, hypertension, TCF7L2 genetic risk marker, medication, activity, diet | Synthetic EHR ([`src/synth_ehr.py`](src/synth_ehr.py)); Synthea importer ([`src/import_synthea.py`](src/import_synthea.py), unit-tested on mock Synthea-layout CSVs) |
| **Dynamic / real-time** | CGM glucose (5 min, with sensor dropouts), heart rate, HRV (RMSSD), steps, sleep stages, meal diary (85% logging rate) | Physiology-based simulator ([`src/simulate_sensors.py`](src/simulate_sensors.py)) |

How credible is the synthetic data? Glycaemic metrics of the simulated cohort against the international consensus on CGM targets:

{GLT}

The simulated cohort is **better controlled and less variable than typical real-world Indian T2D** (see [Data card](docs/DATA_CARD.md)); absolute errors are therefore likely optimistic.

![AGP](results/figures/agp_cohort.png)

## 6. Technical stack

- **Language:** Python 3.10+
- **Data and ML:** NumPy, pandas, scikit-learn (`HistGradientBoosting`, ridge, extra-trees, MLP for comparison), joblib
- **Visualisation and app:** matplotlib, Plotly, Streamlit
- **Synthetic data:** custom EHR generator + physiology simulator; Synthea-compatible importer
- **Quality:** {NTESTS} pytest tests, GitHub Actions CI, Dockerfile, one-command reproducible pipeline (`python -m src.train`)

## 7. AI/ML model and framework details

**Two-layer twin** ([`src/twin.py`](src/twin.py)):

1. **Learned layer** (gradient-boosted trees on {META['features'].__len__()} fused features, see [`src/features.py`](src/features.py))
   - *Forecast models* predict the **change** in glucose over 30 / 60 / 120 min and add it to the current reading.
   - *Uncertainty band:* 10% and 90% quantile models ({f(R['fusion_band_80pct_coverage'], 0)}% empirical coverage for a nominal 80%).
   - *Spike-onset classifier* (glucose <= 180 now, will it exceed 180 in 2 h?) and *hypoglycaemia classifier* (class-weighted; rare event); thresholds tuned on validation patients.
   - *Modality-dropout training:* wearables and the meal diary are randomly masked during training so the model degrades gracefully when a stream disappears.
2. **Mechanistic layer** (personalised physiology for what-if): a per-patient **insulin-sensitivity index** from HbA1c, BMI, age, duration, TCF7L2 alleles, medication and activity drives gamma-shaped meal-response and decaying exercise kernels.
3. **Explanation layer:** each feature group is neutralised to its training median and the shift in forecast and risk is reported.

**Evaluation protocol:** 60/20/20 **patient-level split stratified by medication class** ({dsn[0]} / {dsn[1]} / {dsn[2]} patients); the first 24 h of each series is history only; targets use the noise-free physiological glucose while features see only the observed CGM (with gaps); all CIs resample whole patients.

## 8. Results

All numbers are on the **{dsn[2]} held-out test patients**; regenerate everything with `python -m src.train`. Full detail and methodology: [Technical report](docs/TECHNICAL_REPORT.md).

### 8.1 Two-hour forecast (full CGM history)

{ABL}

![ablation](results/figures/ablation_mae.png)

### 8.2 Cold start: newly onboarded patient (< 1 h of CGM, no sleep history)

{COLD}

![cold](results/figures/cold_start_mae.png)

### 8.3 Spike-onset alert (glucose now <= 180 mg/dL)

Onset prevalence in the test set: {f(D['spike_onset_prevalence_test_pct'], 0)}%.

{CLS}

Fusion vs CGM-only ROC-AUC gain: {f(auc_gain['mean'], 3)} (CI [{f(auc_gain['ci95'][0], 3)}, {f(auc_gain['ci95'][1], 3)}]), {sig_auc}.

### 8.4 Alert burden and operating points

A clinician-usable alert must balance sensitivity against alert fatigue. At the tuned threshold ({tuned['threshold']:.2f}) the twin flags {f(tuned['detected_pct'], 1)}% of excursions (median warning {f(tuned['median_lead_min'], 0)} min, capped by the 2 h look-back) but raises {f(tuned['alert_episodes_per_patient_day'])} alert episodes per patient-day, of which {f(tuned['false_alert_share_pct'], 0)}% are not followed by a spike. Raising the threshold trades detection for fewer alerts:

{OPT}

![operating points](results/figures/operating_points.png)

### 8.5 Hypoglycaemia alert (rare event)

Test set: {D['hypo_positive_rows_test']} positive rows from {D['hypo_positive_patients_test']} patients ({D['insulin_patients_test']} insulin users). Fusion ROC-AUC **{f(HY['fusion']['roc_auc'], 3)}** (CI {f(BO['hypo_auc']['fusion']['ci95'][0], 3)}-{f(BO['hypo_auc']['fusion']['ci95'][1], 3)}) vs CGM-only {f(HY['cgm_only']['roc_auc'], 3)}; at the recall-oriented threshold, recall {f(HY['fusion']['recall'] * 100, 0)}% at precision {f(HY['fusion']['precision'] * 100, 0)}% (PR-AUC {f(HY['fusion']['pr_auc'], 2)} against a {f(HY['fusion']['prevalence'] * 100, 1)}% base rate). {HYPO_TXT} {HYPO_THR}Precision is low, so this is a screening aid, not a diagnosis.

### 8.6 Multi-horizon forecasts

{HZT}

### 8.7 Model families

{FAMT}

{best_fam} is the best on this metric; gradient boosting is within {f(FAM['HistGradientBoosting (ours)']['mae'] - FAM[best_fam]['mae'], 2)} mg/dL of it and was chosen for native missing-value handling, speed and quantile support.

### 8.8 Subgroups

{SGT}

{SUBTXT}

![subgroups](results/figures/subgroup_mae.png)

### 8.9 Robustness (models not retrained for each scenario)

{RBT}

{RBN}

![robustness](results/figures/robustness_mae.png)

### 8.10 Distribution shift and cross-validation

An unseen cohort of {SH['cohort']['patients']} patients, older ({f(SH['cohort']['mean_age'], 0)} vs {f(SH['cohort']['train_mean_age'], 0)} y), heavier (BMI {f(SH['cohort']['mean_bmi'])} vs {f(SH['cohort']['train_mean_bmi'])}), longer disease duration ({f(SH['cohort']['mean_duration_y'])} vs {f(SH['cohort']['train_mean_duration_y'])} y): persistence MAE {f(SH['persistence_mae'])}, CGM only {f(SH['cgm_only_mae'])}, **fusion {f(SH['fusion_mae'])} mg/dL**, spike AUC {f(SH['fusion_spike_auc'], 3)}.

5-fold patient-level cross-validation (whole patients held out per fold):

{CVT}

### 8.11 What drives the forecast

Top features: {TOPF}.

{GRP}

![groups](results/figures/group_importance.png)

### 8.12 Honest reading

- With a full day of CGM history the **EHR and wearables add a real but modest gain** ({f(gain['mean'], 2)} mg/dL, {sig_mae}); glucose history already reveals most of each patient's state.
- The EHR matters most at **cold start** ({cut_cold}% error reduction). For hypoglycaemia, {HYPO_TXT[0].lower() + HYPO_TXT[1:]}
- Wearables add the least in this simulation, partly because the simulator routes their effect (exercise) through steps that the CGM already reflects; on real data their value must be re-measured.
- A plain fusion model **fails badly when wearables vanish** ({f(rb_w['fusion_plain_mae'])} vs CGM-only {f(rb_w['cgm_only_mae'])}); modality-dropout training fixes that ({f(rb_w['fusion_mae'])}).
- Alerts are sensitive but **noisy at the default threshold**; the operating-point table makes the trade-off explicit.

![forecast](results/figures/forecast_example.png)

## 9. Clinician dashboard (working app)

```bash
streamlit run app/dashboard.py
```

- **Twin view:** glucose, +30/+60/+120 min forecast with 80% band, hyper/hypo tiles, "reveal what actually happened", **why-this-forecast** panel, what-if planner, clinical summary.
- **Cohort board:** all demo patients ranked by risk at the current replay time.
- **Data fusion:** the EHR profile, wearable streams, the exact fused feature vector and stream health.
- **Data-quality simulator:** switch off wearables, the meal diary or the CGM and watch the twin degrade gracefully and say so.
- **Model performance:** accuracy, alerts and operating points, subgroups, robustness and shift, explainability, cross-validation.
- **Safety and limits.**

## 10. Demo video

**Unlisted YouTube video (about 20 minutes):** _PASTE LINK HERE_

Script and timestamps: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## 11. Architecture diagram and presentation

- Architecture (2 pages: system, and ML pipeline + evaluation): [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf); PNGs [`docs/architecture_diagram.png`](docs/architecture_diagram.png), [`docs/architecture_pipeline.png`](docs/architecture_pipeline.png)
- Presentation: [`docs/DiaTwin_Presentation.pptx`](docs/DiaTwin_Presentation.pptx) / [`docs/DiaTwin_Presentation.pdf`](docs/DiaTwin_Presentation.pdf)

![pipeline](docs/architecture_pipeline.png)

## 12. Data and sandbox-rule compliance

The challenge restricts teams to anonymised, open-source or synthetic data because of the DPDP Act and HIPAA. **This project uses no real patient data.**

- **EHRs:** synthetic generator in `src/synth_ehr.py` (Synthea-compatible schema). `src/import_synthea.py` maps Synthea CSV exports onto that schema and is unit-tested on mock CSVs in Synthea's layout; it has not been run on a full Synthea export in this submission.
- **Sensor data:** generated by our own simulator, mimicking CGM, Apple Health and Google Fit style streams (5-minute resolution).

## 13. Open-source license

Released under the **MIT License** ([LICENSE](LICENSE)). Third-party libraries (NumPy, pandas, scikit-learn, matplotlib, Plotly, Streamlit) are used under their own permissive licenses.

## 14. How to run

```bash
git clone <this repository>
cd DiaTwin_IMT-Ghaziabad
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                     # {NTESTS} tests
python -m src.train           # regenerate data, retrain, full evaluation suite (~9 min on one CPU core)
streamlit run app/dashboard.py
# or: docker build -t diatwin . && docker run -p 8501:8501 diatwin
```

Pretrained models are committed in `models/` so the dashboard runs immediately. If you use a scikit-learn version other than the pinned one, run `python -m src.train` once to rebuild them.

## 15. Repository structure

```
src/
  synth_ehr.py          synthetic EHR generator (+ cohort-shift parameters)
  import_synthea.py     Synthea CSV importer
  simulate_sensors.py   physiology simulator: CGM / HR / HRV / steps / sleep / meals / drug effect / dropouts
  features.py           fusion of static + dynamic streams, feature sets, multi-horizon targets
  train.py              training + 12-part evaluation suite -> models, metrics.json, figures
  evaluation.py         bootstrap CIs, calibration, alert burden, subgroups, perturbations, glycaemic metrics
  twin.py               DigitalTwin: ingest, forecast, alerts, explain, what-if, data quality
  figures.py            result figures
app/dashboard.py        Streamlit clinician dashboard
data/                   EHR table + 12 demo patients (sensor streams, gz)
models/                 trained models + meta.json
results/                metrics.json, figures
docs/                   report, model card, data card, architecture, presentation, demo script, build scripts
tests/                  pytest suite ({NTESTS} tests)
.github/workflows/      CI
```

## 16. Limitations and next steps

- **Synthetic data only.** Absolute accuracy figures are properties of our simulator, not clinical claims; the simulated CGM variability (CV {f(GL['cv_pct'], 0)}%) is lower than typical real-world T2D, so real errors will be larger. The transferable results are the model ranking, the cold-start effect, the missing-stream robustness and the evaluation methodology.
- **The what-if layer shares assumptions with the simulator,** so its real-world fidelity is untested.
- **Hypoglycaemia** is evaluated on few events from few insulin users ({D['hypo_positive_patients_test']} test patients); treat it as a proof of concept.
- **Alert fatigue:** the default alert threshold is noisy ({f(tuned['false_alert_share_pct'], 0)}% false-alert share); deployment would need threshold tuning with clinicians.
- Unlogged meals remain the main source of residual error.
- **Next:** validate on real anonymised CGM + EHR data, add medication-timing inputs, run a clinician usability study, monitor drift, and review the regulatory pathway for software as a medical device before any clinical use.

## 17. Market context

<!-- TODO (team): add 3-5 healthcare/diabetes startups from the ValleyNxt Ventures Tracxn screen here: name, what they do, what they do not do. -->

Common approaches today: CGM apps show glucose history and trends; care programmes provide coaching and logging; hospital analytics produce periodic risk scores. DiaTwin targets the gap between them: a **personalised, forward-looking, explainable, simulatable** model per patient. (Our assessment, not an exhaustive market survey.)

## Disclaimer

DiaTwin is a research prototype built on synthetic data. It is **not a medical device** and must not be used for diagnosis or treatment decisions.
"""
(ROOT / "README.md").write_text(README)

# ====================================================================== TECHNICAL REPORT
REPORT = f"""# DiaTwin technical report

Digital Twin Challenge 2026, Phase 1. This report documents the clinical rationale, data, methods, evaluation and limits of the DiaTwin prototype. All numbers are generated from `results/metrics.json` (one run of `python -m src.train`).

## 1. Executive summary

DiaTwin is a per-patient digital twin for Type 2 Diabetes. It fuses a static EHR with dynamic CGM, wearable and meal-diary streams to (i) forecast glucose 30, 60 and 120 minutes ahead with an uncertainty band, (ii) warn of hyperglycaemic excursions and hypoglycaemia, (iii) explain each forecast, and (iv) simulate what-if choices. On {dsn[2]} held-out synthetic patients the 2-hour forecast error is **{f(R['fusion']['mae'])} mg/dL** versus {f(R['persistence']['mae'])} for persistence (-{cut_persist}%). Fusing the EHR and wearables improves on CGM alone by {f(gain['mean'], 2)} mg/dL ({sig_mae}) and by **{cut_cold}% at cold start**. The model remains accurate when inputs degrade, provided it is trained with modality dropout. All results are on synthetic data and are not clinical claims.

## 2. Clinical background and rationale

- **Burden.** ICMR-INDIAB estimates 101 million Indians with diabetes and 136 million with prediabetes (Anjana et al., *Lancet Diabetes Endocrinol* 2023).
- **Why a two-hour horizon.** Post-meal glucose excursions peak roughly 45-90 minutes after eating and resolve within about three hours, so a two-hour warning leaves time for a clinically meaningful action (a walk, a smaller portion, a dose adjustment).
- **Why glucose targets are expressed as time in range.** International consensus on CGM metrics (Battelino et al., *Diabetes Care* 2019) defines time in range as 70-180 mg/dL, with time-below-range and time-above-range reported separately. DiaTwin uses the same thresholds (70 and 180 mg/dL) for its two adverse events.
- **Why a digital twin rather than a classifier.** A twin keeps a persistent, personalised state, supports counterfactual questions and degrades visibly when inputs break; a static risk score does none of these.

## 3. Problem formulation

Let g(t) be the physiological glucose and g_obs(t) what the CGM reports (with gaps). At each 5-minute step the twin sees the history of g_obs, the wearable and meal-diary streams, and the static EHR vector x. It estimates:

1. **Forecast:** g(t+h) - g_obs(t) for h in {{30, 60, 120}} min (the model predicts the change; the current reading is added back).
2. **Spike onset:** among rows with g_obs(t) <= 180, whether max g over (t, t+120 min] exceeds 180 mg/dL. Restricting to rows that are not already high avoids rewarding trivial persistence.
3. **Hypoglycaemia onset:** among rows with g_obs(t) >= 70, whether min g over (t, t+120 min] falls below 70 mg/dL.

Targets use the noise-free physiological glucose, so sensor noise and dropouts are a property of the inputs only.

## 4. Data

### 4.1 Synthetic EHR

{D['patients']} patients; age ~ N(52, 11), BMI ~ N(27.5, 4.2), disease duration ~ Gamma(2.2, 3.0) years, TCF7L2 risk alleles in {{0, 1, 2}} with probabilities (0.45, 0.43, 0.12). HbA1c = 6.5 + 0.045 (BMI - 25) + 0.06 duration + 0.20 alleles + N(0, 0.7). Fasting glucose = 0.82 x eAG + noise, with eAG = 28.7 HbA1c - 46.7 (the ADA conversion). eGFR, LDL, hypertension, family history, activity level, diet and sleep habit complete the record. Medication escalates with HbA1c (none / metformin, then + sulfonylurea, then insulin). Mix: {", ".join(f"{k} {f(v, 0)}%" for k, v in D['medication_mix_pct'].items())}.

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

The drug term is what creates hypoglycaemia: a dose is taken for a planned meal that may be skipped (about 4% of meals plus the planned slot) or eaten smaller, and doses are sometimes taken for skipped meals (50%). Heart rate and HRV follow activity, sleep, stress, age and BMI. **Imperfections** are modelled deliberately: 85% of meals are logged in the diary, and CGM dropouts remove short segments (about {f(GL['cgm_dropout_pct'], 1)}% of readings).

### 4.3 Realism check

{GLT}

The cohort sits comfortably in the consensus-recommended range, which means it is **better controlled and less variable** than typical real-world Indian T2D (population mean HbA1c is higher in practice, and real CGM CV is usually above the {f(GL['cv_pct'], 0)}% simulated here). Absolute errors will therefore be optimistic. The distribution-shift cohort (Section 7.9) partly addresses this by testing an older, heavier, longer-duration population.

### 4.4 Synthea compatibility

`src/import_synthea.py` maps Synthea's `patients.csv`, `conditions.csv` and `observations.csv` (LOINC 4548-4 HbA1c, 39156-5 BMI, 2339-0 glucose, 33914-3 eGFR, 18262-6 LDL) onto the same schema, then fills fields Synthea does not model (genetic marker, lifestyle) from the generator's distributions. It is unit-tested on mock CSVs in Synthea's layout and the output is verified to flow through the simulator and feature pipeline.

## 5. Features ({len(META['features'])} fused features)

| Group | Count | Examples |
|---|---|---|
| CGM history and trend | {len(META['groups']['Glucose history & trend (CGM)'])} | current glucose, lags (5-120 min), 15/30/60-min deltas, 1 h and 3 h rolling stats, 24 h mean, CGM gap, hour of day (sin, cos) |
| Meal diary | {len(META['groups']['Meals (diary)'])} | logged carbs in last 1 h / 3 h, minutes since last meal |
| Wearables | {len(META['groups']['Activity, HR, HRV, sleep (wearables)'])} | steps (15 min, 1 h, 3 h), heart rate, HRV, sleep hours (24 h), HRV during sleep |
| EHR profile | {len(META['groups']['EHR profile'])} | age, sex, BMI, duration, HbA1c, fasting glucose, eGFR, LDL, hypertension, family history, TCF7L2 alleles, medication flags, activity, diet |

Short CGM gaps (up to 30 minutes) are bridged by carrying the last reading forward; longer gaps leave the value missing and the model handles it natively. Missing wearables stay missing (not zero), so a switched-off device is distinguishable from a sedentary one.

## 6. Models

- **Learned layer.** `HistGradientBoostingRegressor` / `Classifier` (300 iterations max, learning rate 0.06, 31 leaves, L2 = 1.0, early stopping on 10% internal validation). Native handling of missing values makes it a natural fit for streams that drop out.
- **Delta target.** Predicting g(t+h) - g_obs(t) rather than g(t+h) removes the dominant persistence component and lets the trees spend capacity on the dynamics.
- **Uncertainty.** Two quantile regressors (10%, 90%) give an 80% band; empirical coverage is {f(R['fusion_band_80pct_coverage'], 0)}%.
- **Classifiers.** Spike onset uses the tuned threshold ({META['spike_threshold_prob']:.2f}, maximising F1 on validation patients). Hypoglycaemia uses `class_weight='balanced'` and a recall-oriented threshold ({META['hypo_threshold_prob']:.2f}, maximising F2), so its output is a ranking score rather than a calibrated probability.
- **Modality-dropout training.** In training rows, wearables are masked in 15% of rows, the meal diary is blanked in 15% and both in 5%. This is the difference between a model that collapses when a stream disappears and one that falls back gracefully (Section 7.8).
- **Mechanistic layer.** Reuses the simulator's kernels with the patient's insulin-sensitivity index to produce counterfactual 3-hour curves: eat, eat half, eat then walk, skip. It starts from the current reading and adds the residual response of meals already logged.
- **Explanation layer.** For each feature group the group is replaced by its training median and the change in the predicted 2-hour delta and spike risk is reported (positive = pushes glucose up).

## 7. Evaluation and results

**Protocol.** 60/20/20 split by patient, stratified by medication class so that rare insulin users appear in every split ({dsn[0]} / {dsn[1]} / {dsn[2]} patients). The first 24 h of each series is history only. Training and test rows are sampled every 15 minutes. All confidence intervals are patient-level bootstrap (whole patients resampled).

### 7.1 Ablation: forecast

{ABL}

Fusion vs CGM-only: paired MAE gain {f(gain['mean'], 2)} mg/dL, 95% CI {ci(gain)} ({sig_mae}).

### 7.2 Ablation: spike-onset alert

Prevalence in the test set {f(D['spike_onset_prevalence_test_pct'], 0)}%.

{CLS}

AUC gain of fusion over CGM-only {f(auc_gain['mean'], 3)} (CI [{f(auc_gain['ci95'][0], 3)}, {f(auc_gain['ci95'][1], 3)}]), {sig_auc}. Calibration of the final spike classifier: Brier {f(M['calibration']['spike']['brier'], 3)}, ECE {f(M['calibration']['spike']['ece'], 3)}.

![calibration](../results/figures/calibration_spike.png)

### 7.3 Cold start

{COLD}

### 7.4 Model families

{FAMT}

### 7.5 Multi-horizon

{HZT}

![horizon](../results/figures/horizon_mae.png)

### 7.6 Alert burden, lead time and operating points

Alert episodes are runs of consecutive alerting rows; an episode is *false* if glucose does not cross 180 mg/dL within the following 2 hours; an excursion is *flagged* if an alert fired in the preceding 2 hours, and its lead time is the time from the first such alert to the crossing (capped at 120 minutes by the look-back).

{OPT}

![lead](../results/figures/alert_lead_time.png)

### 7.7 Hypoglycaemia

{D['hypo_positive_rows_test']} positive rows from {D['hypo_positive_patients_test']} test patients ({D['insulin_patients_test']} on insulin). Fusion ROC-AUC {f(HY['fusion']['roc_auc'], 3)} (CI {ci3(BO['hypo_auc']['fusion'])}) vs CGM-only {f(HY['cgm_only']['roc_auc'], 3)}; fusion recall {f(HY['fusion']['recall'] * 100, 0)}% at precision {f(HY['fusion']['precision'] * 100, 0)}%, PR-AUC {f(HY['fusion']['pr_auc'], 2)} against a base rate of {f(HY['fusion']['prevalence'] * 100, 2)}%. {HYPO_TXT} {HYPO_THR}The low precision and the small number of positive patients mean this is a proof of concept.

### 7.8 Robustness

Scenarios degrade the observed streams of the test patients; the targets are unchanged and models are not retrained.

{RBT}

Plain fusion training fails when wearables disappear ({f(rb_w['fusion_plain_mae'])} mg/dL, worse than CGM only at {f(rb_w['cgm_only_mae'])}); modality-dropout training restores it ({f(rb_w['fusion_mae'])}). {RBN}

![robust](../results/figures/robustness_mae.png)

### 7.9 Distribution shift

Unseen cohort ({SH['cohort']['patients']} patients; age {f(SH['cohort']['mean_age'], 0)} vs {f(SH['cohort']['train_mean_age'], 0)}, BMI {f(SH['cohort']['mean_bmi'])} vs {f(SH['cohort']['train_mean_bmi'])}, duration {f(SH['cohort']['mean_duration_y'])} vs {f(SH['cohort']['train_mean_duration_y'])} y, HbA1c {f(SH['cohort']['mean_hba1c'])} vs {f(SH['cohort']['train_mean_hba1c'])}): persistence {f(SH['persistence_mae'])}, CGM only {f(SH['cgm_only_mae'])}, fusion {f(SH['fusion_mae'])} mg/dL, spike AUC {f(SH['fusion_spike_auc'], 3)}. The ranking is preserved under shift; absolute error rises because this population is intrinsically harder to forecast.

### 7.10 Subgroups

{SGT}

{SUBTXT}

![sub](../results/figures/subgroup_mae.png)

### 7.11 Cross-validation

{CVT}

### 7.12 Importance

Top features by permutation importance: {TOPF}. By group:

{GRP}

![imp](../results/figures/feature_importance.png)

## 8. Safety, ethics and regulation

- **Human in the loop.** DiaTwin is decision support for a clinician. It never doses, and it makes its uncertainty visible (80% band, risk tiers, data-quality warnings).
- **Alert fatigue.** The default threshold is sensitive and noisy; Section 7.6 exposes the trade-off so clinicians can choose an operating point. A deployed system should add snooze logic and per-patient thresholds.
- **Fairness.** Performance is reported by HbA1c band, medication, age, sex and activity (Section 7.10). Real-world fairness analysis needs real demographics, including region, language and device access.
- **Failure behaviour.** CGM gaps, offline wearables and unlogged meals are detected and reported, and the model was trained to cope with them. The dashboard's data-quality simulator demonstrates this.
- **Privacy.** No real patient data is used; DPDP Act and HIPAA constraints are satisfied by construction. A real deployment would require consent, data minimisation and on-device or in-region processing.
- **Regulation.** A tool that influences treatment decisions would likely be regulated as software as a medical device; the applicable pathway must be assessed with qualified advisers before any clinical use.

## 9. Limitations and threats to validity

1. **Synthetic data.** Everything shares the simulator's assumptions. The simulated CGM variability (CV {f(GL['cv_pct'], 0)}%) and control (TIR {f(GL['tir_70_180_pct'], 0)}%) are more benign than typical real-world T2D, so real errors will be larger.
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

`python -m src.train` regenerates every artefact from seeds (EHR seed 42, simulator seed 7, split seed 0). `pytest -q` runs {NTESTS} tests, including physiology monotonicity, target-leakage checks, the Synthea importer, alert-metric arithmetic and graceful degradation. CI runs the tests and a dashboard smoke test on every push.

## 12. References

1. Anjana RM et al. Metabolic non-communicable disease health report of India: the ICMR-INDIAB cross-sectional study. *Lancet Diabetes & Endocrinology*, 2023.
2. Battelino T et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. *Diabetes Care*, 2019.
3. Bergenstal RM et al. Glucose management indicator (GMI). *Diabetes Care*, 2018.
4. Walonoski J et al. Synthea: an approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. *JAMIA*, 2018.
5. Johnson AEW et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 2023.
6. Pedregosa F et al. Scikit-learn: machine learning in Python. *JMLR*, 2011.
"""
(ROOT / "docs" / "TECHNICAL_REPORT.md").write_text(REPORT)

# ====================================================================== MODEL CARD
MODEL_CARD = f"""# Model card: DiaTwin glucose forecaster and alerts

## Model details
- **Developers:** Team DiaTwin, IMT Ghaziabad. Version 1.0 (Digital Twin Challenge 2026 prototype). License: MIT.
- **Type:** gradient-boosted decision trees (scikit-learn `HistGradientBoosting`), seven models: 30/60/120-minute delta regressors, 10% and 90% quantile regressors (2 h), spike-onset classifier, hypoglycaemia classifier. Plus a mechanistic what-if layer and a feature-group explanation layer.
- **Inputs:** {len(META['features'])} features fusing static EHR fields with CGM, wearable and meal-diary streams (see `models/meta.json`).
- **Outputs:** glucose at +30/+60/+120 min, 80% band at 2 h, spike-onset risk, hypoglycaemia risk score, feature-group explanation.

## Intended use
Research and demonstration of a digital-twin concept for Type 2 Diabetes: decision support shown to a clinician on synthetic patients. **Not for diagnosis, dosing or any clinical decision.**

## Out-of-scope uses
Real patients; Type 1 diabetes, gestational or paediatric diabetes; automated insulin delivery; unsupervised alerts to patients.

## Training and evaluation data
Fully synthetic: {D['patients']} patients x {D['days_per_patient']} days at 5-minute resolution ({D['sensor_rows']:,} readings), see the [Data card](DATA_CARD.md). Patient-level split {dsn[0]} / {dsn[1]} / {dsn[2]}, stratified by medication class; an independent shifted cohort of {SH['cohort']['patients']} patients was used for the distribution-shift test.

## Performance (test patients)
| Task | Result |
|---|---|
| 2-hour forecast MAE | {f(R['fusion']['mae'])} mg/dL (CI {ci(BO['mae']['fusion'])}); persistence {f(R['persistence']['mae'])} |
| 30 / 60-minute MAE | {f(HZ['30']['fusion']['mae'])} / {f(HZ['60']['fusion']['mae'])} mg/dL |
| Cold-start 2-hour MAE | {f(CS['cold_fusion']['mae'])} mg/dL (CGM only {f(CS['cold_cgm']['mae'])}) |
| Spike-onset ROC-AUC / PR-AUC | {f(C['fusion']['roc_auc'], 3)} / {f(C['fusion']['pr_auc'], 3)}; ECE {f(M['calibration']['spike']['ece'], 3)} |
| Spike alert at tuned threshold | recall {f(C['fusion']['recall'] * 100, 0)}%, precision {f(C['fusion']['precision'] * 100, 0)}%; {f(tuned['alert_episodes_per_patient_day'])} episodes/patient-day |
| Hypoglycaemia ROC-AUC | {f(HY['fusion']['roc_auc'], 3)} (CGM only {f(HY['cgm_only']['roc_auc'], 3)}; rare event, {D['hypo_positive_patients_test']} positive patients) |
| 5-fold CV MAE | {f(cvf['mean'], 2)} +/- {f(cvf['sd'], 2)} (CGM only {f(cvc['mean'], 2)} +/- {f(cvc['sd'], 2)}) |

## Factors and subgroup analysis
Performance by HbA1c band, medication, age, sex and activity is in the [Technical report](TECHNICAL_REPORT.md) (Section 7.10). {SUBTXT}

## Robustness
Tested without retraining under CGM noise, 20% CGM dropout, no meal diary and wearables offline. Modality-dropout training is essential: with wearables offline MAE is {f(rb_w['fusion_mae'])} mg/dL against {f(rb_w['fusion_plain_mae'])} for plain training.

## Ethical considerations and risks
- Alert fatigue (see operating points), false reassurance when a stream silently fails (mitigated by data-quality warnings), and over-trust in forecasts made on synthetic data.
- The model has never seen real patients; subgroup fairness on real populations is unknown.
- The hypoglycaemia output is a ranking score, not a calibrated probability.

## Caveats and recommendations
Treat all accuracy figures as properties of the simulator. Before any clinical use: validate on real anonymised data, recalibrate thresholds with clinicians, monitor drift, and complete a regulatory review.
"""
(ROOT / "docs" / "MODEL_CARD.md").write_text(MODEL_CARD)

# ====================================================================== DATA CARD
DATA_CARD = f"""# Data card: DiaTwin synthetic cohort

## Overview
Fully synthetic, generated on demand by `python -m src.train` from fixed seeds. Contains **no real patient data and no personal identifiers**. Committed files: `data/ehr.csv` (all patients) and `data/demo_*` (12 held-out demo patients).

## Composition
| | |
|---|---|
| Patients | {D['patients']} (+ {SH['cohort']['patients']} in a shifted evaluation cohort) |
| Duration | {D['days_per_patient']} days per patient |
| Resolution | 5 minutes |
| Sensor rows | {D['sensor_rows']:,} |
| Medication mix | {", ".join(f"{k} {f(v, 0)}%" for k, v in D['medication_mix_pct'].items())} |

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
{GLT}

## Known biases and limitations
- **Too benign.** TIR {f(GL['tir_70_180_pct'], 0)}% and CV {f(GL['cv_pct'], 0)}% are better and tighter than typical real-world T2D, especially in India where mean HbA1c is higher.
- **Simplified physiology.** One meal kernel shape, no fat/protein effects, no illness, no alcohol, no insulin-dose timing errors beyond a mismatch factor.
- **Regular routines.** Meal times are tightly scheduled, so time of day is a strong predictor.
- **Few insulin users** ({f(D['medication_mix_pct'].get('insulin', 0), 0)}% of the cohort), so hypoglycaemia events are rare and concentrated.
- **No demographic diversity beyond the modelled variables** (no region, language, device type).

## Intended use
Prototyping and evaluating digital-twin methods under the Digital Twin Challenge sandbox rules. Not for estimating real-world clinical performance.

## Privacy and compliance
No real data, so DPDP Act and HIPAA restrictions are met by construction.

## Maintenance and license
Regenerate with `python -m src.train`. MIT License.
"""
(ROOT / "docs" / "DATA_CARD.md").write_text(DATA_CARD)
print("docs written")
