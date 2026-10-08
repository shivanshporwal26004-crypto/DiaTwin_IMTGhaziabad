"""End-to-end pipeline: generate data -> features -> train -> rigorous evaluation -> artefacts.

    python -m src.train            # full run (~8-12 min on a laptop CPU)

Evaluation suite (all patient-level, nothing leaks across patients):
    1  feature-set ablation (CGM / +wearables / +EHR / fusion) for the 2 h forecast, spike-onset and hypoglycaemia alerts
    2  cold-start test (new patient, < 1 h of CGM)
    3  model-family comparison (ridge, extra-trees, MLP, gradient boosting)
    4  multi-horizon forecasts (30 / 60 / 120 min)
    5  uncertainty band coverage, calibration (Brier, ECE)
    6  patient-level bootstrap 95% CIs and paired gains
    7  alert burden, false-alert share and lead time
    8  subgroup analysis (HbA1c, medication, age, sex, activity)
    9  robustness (CGM noise, CGM dropout, no meal log, wearables offline)
   10  distribution shift (older / heavier / longer-duration cohort)
   11  5-fold patient-level cross-validation
   12  permutation importance (feature and feature-group)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluation as ev
from .config import HORIZON_MIN, HYPO_MGDL, SPIKE_MGDL
from .features import COLD_SETS, CGM, EHR, FEATURE_SETS, SENSOR, build_features
from .simulate_sensors import simulate_cohort
from .synth_ehr import generate_ehr

ROOT = Path(__file__).resolve().parent.parent
GBM = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
MEALS = ["carbs_1h", "carbs_3h", "min_since_meal"]
GROUPS = {"Glucose history & trend (CGM)": CGM, "Meals (diary)": MEALS,
          "Activity, HR, HRV, sleep (wearables)": [c for c in SENSOR if c not in MEALS],
          "EHR profile": EHR}


WEAR = [c for c in SENSOR if c not in MEALS]


def mask_modalities(df, seed=0):
    """Modality-dropout augmentation: randomly simulate wearables offline (15%), meal diary unused (15%), both (5%).
    Teaches the fusion model to degrade gracefully instead of failing when a stream disappears."""
    r = np.random.default_rng(seed)
    d = df.copy()
    u = r.random(len(d))
    off_w = (u < 0.15) | ((u >= 0.30) & (u < 0.35))
    off_m = ((u >= 0.15) & (u < 0.30)) | ((u >= 0.30) & (u < 0.35))
    d.loc[off_w, WEAR] = np.nan
    d.loc[off_m, ["carbs_1h", "carbs_3h"]] = 0.0
    d.loc[off_m, "min_since_meal"] = 480.0
    return d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_patients(ehr, seed=0):
    """60/20/20 patient-level split, stratified by medication class (rare insulin users appear in every split)."""
    rng = np.random.default_rng(seed)
    strata = np.where(ehr.medication == "insulin", "insulin", np.where(ehr.medication.str.contains("sulfonylurea"), "sulf", "other"))
    parts = ([], [], [])
    for st in np.unique(strata):
        ids = ehr.patient_id.values[strata == st].copy()
        rng.shuffle(ids)
        n = len(ids)
        a, b = int(.6 * n), int(.8 * n)
        parts[0].extend(ids[:a]); parts[1].extend(ids[a:b]); parts[2].extend(ids[b:])
    return tuple(set(p) for p in parts)


def reg_metrics(y, p):
    err = p - y
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mard_pct": float(np.mean(np.abs(err) / y) * 100),
            "within15mgdl_pct": float(np.mean(np.abs(err) <= 15) * 100),
            "within20_pct": float(np.mean(np.abs(err) / y <= 0.20) * 100)}


def clf_metrics(y, prob, thr):
    pred = prob >= thr
    return {"roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "threshold": float(thr), "prevalence": float(np.mean(y))}


def best_threshold(y, prob, beta=1.0):
    ts = np.linspace(0.02, 0.95, 94)
    return float(ts[int(np.argmax([fbeta_score(y, prob >= t, beta=beta) for t in ts]))])


def fit_delta(cols, tr, target="y_g120", **kw):
    return HistGradientBoostingRegressor(**{**GBM, **kw}).fit(tr[cols], tr[target] - tr.g_now)


def mae_of(d, delta, target="y_g120"):
    return float(np.abs(d.g_now.values + delta - d[target].values).mean())


def eval_models(d, regs, clfs, dO):
    """Compact evaluation used for robustness / shift: MAE (persistence, CGM-only, fusion) + spike AUC."""
    out = {"rows": int(len(d)), "persistence_mae": float(np.abs(d.g_now - d.y_g120).mean())}
    for name in ("cgm_only", "fusion", "fusion_plain"):
        cs = FEATURE_SETS.get(name, FEATURE_SETS["fusion"])
        out[f"{name}_mae"] = mae_of(d, regs[name].predict(d[cs]))
        if dO.y_spike.nunique() > 1:
            out[f"{name}_spike_auc"] = float(roc_auc_score(dO.y_spike, clfs[name].predict_proba(dO[cs])[:, 1]))
    return out


def main(n_patients=400, days=14, seed=7, make_figs=True):
    t0 = time.time()
    for d in ("models", "results/figures", "data"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    M = {"horizon_min": HORIZON_MIN}

    # ------------------------------------------------------------------ data
    log("1/12 generating synthetic EHR + sensor streams")
    ehr = generate_ehr(n_patients, seed=42)
    sensors = simulate_cohort(ehr, days=days, seed=seed)
    ehr.to_csv(ROOT / "data" / "ehr.csv", index=False)
    feats = build_features(sensors, ehr)
    tr_ids, va_ids, te_ids = split_patients(ehr)
    sub = feats["step_idx"] % 3 == 0                                   # 15-minute sampling
    tr, va, te = (feats[feats.patient_id.isin(s) & sub] for s in (tr_ids, va_ids, te_ids))
    onset = lambda d: d[d.g_now <= SPIKE_MGDL]                          # spike *onset*: not yet high
    hypo_rows = lambda d: d[d.g_now >= HYPO_MGDL]                       # hypo onset: not yet low
    tr_o, va_o, te_o = onset(tr), onset(va), onset(te)
    tr_h, va_h, te_h = hypo_rows(tr), hypo_rows(va), hypo_rows(te)
    tr_m = mask_modalities(tr)                                          # augmented training frame for the final fusion model
    tr_o_m, tr_h_m = onset(tr_m), hypo_rows(tr_m)

    M["dataset"] = {
        "patients": int(n_patients), "days_per_patient": days, "sensor_rows": int(len(sensors)),
        "feature_rows_total": int(len(feats)), "n_features_fusion": len(FEATURE_SETS["fusion"]),
        "train_val_test_patients": [len(tr_ids), len(va_ids), len(te_ids)],
        "spike_onset_prevalence_test_pct": float(te_o.y_spike.mean() * 100),
        "hypo_onset_prevalence_test_pct": float(te_h.y_hypo.mean() * 100),
        "hypo_positive_rows_test": int(te_h.y_hypo.sum()),
        "hypo_positive_patients_test": int(te_h[te_h.y_hypo == 1].patient_id.nunique()),
        "insulin_patients_test": int(ehr[ehr.patient_id.isin(te_ids) & (ehr.medication == "insulin")].shape[0]),
        "glycemic": ev.glycemic_metrics(sensors),
        "medication_mix_pct": {k: float(v * 100) for k, v in ehr.medication.value_counts(normalize=True).items()},
    }
    agp = ev.agp_profile(sensors)

    # ------------------------------------------------------------------ 1. ablation
    log("2/12 ablation: forecast + spike alert + hypo alert")
    R = {"persistence": reg_metrics(te.y_g120.values, te.g_now.values)}
    trend = te.g_now.values + np.clip(te.g_d30.fillna(0).values * 2.0, -60, 80)           # linear-trend extrapolation
    R["linear_trend"] = reg_metrics(te.y_g120.values, trend)
    CL, HY = {}, {}
    regs, clfs, hypo_clfs, thr = {}, {}, {}, {}
    delta, prob_o, prob_all, prob_h = {}, {}, {}, {}
    for name, cols in FEATURE_SETS.items():
        T, T_o, T_h = (tr_m, tr_o_m, tr_h_m) if name == "fusion" else (tr, tr_o, tr_h)   # final fusion = modality-dropout trained
        regs[name] = fit_delta(cols, T)
        delta[name] = regs[name].predict(te[cols])
        R[name] = reg_metrics(te.y_g120.values, te.g_now.values + delta[name])
        clfs[name] = HistGradientBoostingClassifier(**GBM).fit(T_o[cols], T_o.y_spike)
        thr[name] = best_threshold(va_o.y_spike.values, clfs[name].predict_proba(va_o[cols])[:, 1])
        prob_o[name] = clfs[name].predict_proba(te_o[cols])[:, 1]
        prob_all[name] = clfs[name].predict_proba(te[cols])[:, 1]
        CL[name] = clf_metrics(te_o.y_spike.values, prob_o[name], thr[name])
        if name in ("cgm_only", "fusion"):
            hypo_clfs[name] = HistGradientBoostingClassifier(**GBM, class_weight="balanced").fit(T_h[cols], T_h.y_hypo)
            th = best_threshold(va_h.y_hypo.values, hypo_clfs[name].predict_proba(va_h[cols])[:, 1], beta=2.0)
            prob_h[name] = hypo_clfs[name].predict_proba(te_h[cols])[:, 1]
            HY[name] = {**clf_metrics(te_h.y_hypo.values, prob_h[name], th), "f2": float(fbeta_score(te_h.y_hypo.values, prob_h[name] >= th, beta=2.0))}
            thr[f"hypo_{name}"] = th
        log(f"   {name:14s} MAE={R[name]['mae']:.2f}  spike-AUC={CL[name]['roc_auc']:.3f}")
    # plain fusion (no modality dropout): used only to show what the augmentation buys under missing data
    fcols = FEATURE_SETS["fusion"]
    regs["fusion_plain"] = fit_delta(fcols, tr)
    clfs["fusion_plain"] = HistGradientBoostingClassifier(**GBM).fit(tr_o[fcols], tr_o.y_spike)
    R["fusion_plain"] = reg_metrics(te.y_g120.values, te.g_now.values + regs["fusion_plain"].predict(te[fcols]))
    CL["fusion_plain"] = clf_metrics(te_o.y_spike.values, clfs["fusion_plain"].predict_proba(te_o[fcols])[:, 1], thr["fusion"])
    M["regression"], M["classification"], M["hypoglycaemia"] = R, CL, HY
    log(f"   hypo fusion AUC={HY['fusion']['roc_auc']:.3f} PR-AUC={HY['fusion']['pr_auc']:.3f}")

    # ------------------------------------------------------------------ 2. cold start
    log("3/12 cold start")
    M["cold_start"] = {}
    for name, cols in COLD_SETS.items():
        m = fit_delta(cols, tr)
        M["cold_start"][name] = reg_metrics(te.y_g120.values, te.g_now.values + m.predict(te[cols]))

    # ------------------------------------------------------------------ 3. model families
    log("4/12 model families")
    cols = FEATURE_SETS["fusion"]
    rng = np.random.default_rng(0)
    trs = tr.iloc[rng.choice(len(tr), min(90_000, len(tr)), replace=False)]
    fam = {"HistGradientBoosting (ours)": delta["fusion"]}
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)).fit(trs[cols], trs.y_g120 - trs.g_now)
    fam["Ridge regression"] = ridge.predict(te[cols])
    et = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=80, min_samples_leaf=5, max_depth=14, n_jobs=1, random_state=0)).fit(trs[cols], trs.y_g120 - trs.g_now)
    fam["Extra-trees (80)"] = et.predict(te[cols])
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=150, random_state=0)).fit(trs[cols], trs.y_g120 - trs.g_now)
    fam["MLP (64-32)"] = mlp.predict(te[cols])
    M["model_families"] = {k: reg_metrics(te.y_g120.values, te.g_now.values + v) for k, v in fam.items()}

    # ------------------------------------------------------------------ 4. multi-horizon
    log("5/12 multi-horizon")
    M["horizons"] = {}
    for h in (30, 60, 120):
        tgt = f"y_g{h}"
        row = {"persistence": reg_metrics(te[tgt].values, te.g_now.values)}
        for name in ("cgm_only", "fusion"):
            dl = delta[name] if h == 120 else fit_delta(FEATURE_SETS[name], tr_m if name == "fusion" else tr, tgt).predict(te[FEATURE_SETS[name]])
            row[name] = reg_metrics(te[tgt].values, te.g_now.values + dl)
        M["horizons"][str(h)] = row
    reg30 = fit_delta(cols, tr_m, "y_g30")
    reg60 = fit_delta(cols, tr_m, "y_g60")

    # ------------------------------------------------------------------ 5. uncertainty + calibration + CIs
    log("6/12 uncertainty band, calibration, bootstrap CIs")
    q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **GBM).fit(tr_m[cols], tr_m.y_g120 - tr_m.g_now)
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **GBM).fit(tr_m[cols], tr_m.y_g120 - tr_m.g_now)
    lo, hi = te.g_now.values + q10.predict(te[cols]), te.g_now.values + q90.predict(te[cols])
    M["regression"]["fusion_band_80pct_coverage"] = float(np.mean((te.y_g120.values >= lo) & (te.y_g120.values <= hi)) * 100)
    M["regression"]["fusion_band_mean_width_mgdl"] = float(np.mean(hi - lo))
    M["calibration"] = {"spike": ev.calibration(te_o.y_spike.values, prob_o["fusion"]),
                        "hypo_note": "the hypoglycaemia classifier uses class_weight='balanced', so its output is a ranking score, not a calibrated probability"}
    M["bootstrap"] = {
        "mae": ev.patient_bootstrap_mae(te, {"persistence": None, "cgm_only": delta["cgm_only"], "fusion": delta["fusion"]}),
        "mae_fusion_vs_cgm": ev.patient_bootstrap_mae(te, {"cgm_only": delta["cgm_only"], "fusion": delta["fusion"]}),
        "spike_auc": {k: ev.patient_bootstrap_auc(te_o, prob_o[k]) for k in ("cgm_only", "fusion")},
        "spike_auc_gain_fusion_vs_cgm": ev.paired_auc_gain(te_o, prob_o["cgm_only"], prob_o["fusion"]),
        "hypo_auc": {k: ev.patient_bootstrap_auc(te_h, prob_h[k], "y_hypo") for k in ("cgm_only", "fusion")},
    }

    # ------------------------------------------------------------------ 6. alerts + subgroups
    log("7/12 alert burden + subgroups")
    M["alerts"] = {k: ev.alert_metrics(te, prob_all[k], thr[k]) for k in ("cgm_only", "fusion")}
    ops = []
    for t in sorted({0.2, round(thr["fusion"], 2), 0.5, 0.7, 0.85}):
        a = ev.alert_metrics(te, prob_all["fusion"], t)
        pred_o = prob_o["fusion"] >= t
        ops.append({"threshold": t, "tuned": bool(abs(t - round(thr["fusion"], 2)) < 1e-9),
                    "alert_episodes_per_patient_day": a["alert_episodes_per_patient_day"], "false_alert_share_pct": a["false_alert_share_pct"],
                    "detected_pct": a["detected_pct"], "detected_with_30min_lead_pct": a["detected_with_30min_lead_pct"],
                    "median_lead_min": a["median_lead_min"], "row_precision": float(precision_score(te_o.y_spike, pred_o, zero_division=0)),
                    "row_recall": float(recall_score(te_o.y_spike, pred_o))})
    M["operating_points"] = ops
    M["subgroups"] = ev.subgroup_table(te, {"cgm_only": delta["cgm_only"], "fusion": delta["fusion"]}, te_o, prob_o["fusion"])

    # ------------------------------------------------------------------ 7. importance
    log("8/12 permutation importance")
    samp = te.sample(min(5000, len(te)), random_state=0)
    ytrue = samp.y_g120 - samp.g_now
    base = np.abs(regs["fusion"].predict(samp[cols]) - ytrue).mean()
    prng = np.random.default_rng(1)

    def perm_increase(cs, reps=4):
        inc = []
        for _ in range(reps):
            X = samp[cols].copy()
            X[cs] = X[cs].values[prng.permutation(len(X))]
            inc.append(np.abs(regs["fusion"].predict(X) - ytrue).mean() - base)
        return float(np.mean(inc))
    imp = pd.Series({c: perm_increase([c], 3) for c in cols}).sort_values(ascending=False)
    gimp = pd.Series({g: perm_increase([c for c in cs if c in cols]) for g, cs in GROUPS.items()}).sort_values(ascending=False)
    M["top_features"] = {k: float(v) for k, v in imp.head(15).items()}
    M["group_importance"] = {k: float(v) for k, v in gimp.items()}

    # ------------------------------------------------------------------ 8. robustness
    log("9/12 robustness perturbations")
    te_sens = sensors[sensors.patient_id.isin(te_ids)]
    M["robustness"] = {}
    for kind in ("clean", "cgm_noise_8mgdl", "cgm_dropout_20pct", "no_meal_log", "wearables_offline"):
        f = build_features(ev.perturb(te_sens, kind, seed=3), ehr)
        f = f[f.step_idx % 3 == 0]
        M["robustness"][kind] = eval_models(f, regs, clfs, onset(f))
        log(f"   {kind:20s} fusion={M['robustness'][kind]['fusion_mae']:.2f} plain-fusion={M['robustness'][kind]['fusion_plain_mae']:.2f} cgm-only={M['robustness'][kind]['cgm_only_mae']:.2f}")

    # ------------------------------------------------------------------ 9. distribution shift
    log("10/12 distribution shift cohort")
    ehr_s = generate_ehr(120, seed=99, age_mu=63, bmi_add=2.5, dur_scale=1.5, id_prefix="S")
    sens_s = simulate_cohort(ehr_s, days=days, seed=123)
    f_s = build_features(sens_s, ehr_s)
    f_s = f_s[f_s.step_idx % 3 == 0]
    M["distribution_shift"] = {
        "cohort": {"patients": 120, "mean_age": float(ehr_s.age.mean()), "mean_bmi": float(ehr_s.bmi.mean()),
                   "mean_duration_y": float(ehr_s.duration_years.mean()), "mean_hba1c": float(ehr_s.hba1c.mean()),
                   "train_mean_age": float(ehr.age.mean()), "train_mean_bmi": float(ehr.bmi.mean()),
                   "train_mean_duration_y": float(ehr.duration_years.mean()), "train_mean_hba1c": float(ehr.hba1c.mean())},
        **eval_models(f_s, regs, clfs, onset(f_s))}

    # ------------------------------------------------------------------ 10. cross-validation
    log("11/12 5-fold patient-level cross-validation")
    allf = feats[feats["step_idx"] % 6 == 0]
    cv = {"persistence": [], "cgm_only": [], "fusion": [], "cold_cgm": [], "cold_fusion": []}
    for k, (a, b) in enumerate(GroupKFold(5).split(allf, groups=allf.patient_id)):
        A, B = allf.iloc[a], allf.iloc[b]
        cv["persistence"].append(float(np.abs(B.g_now - B.y_g120).mean()))
        for nm, cs in (("cgm_only", FEATURE_SETS["cgm_only"]), ("fusion", FEATURE_SETS["fusion"]),
                       ("cold_cgm", COLD_SETS["cold_cgm"]), ("cold_fusion", COLD_SETS["cold_fusion"])):
            m = fit_delta(cs, A, max_iter=150)
            cv[nm].append(mae_of(B, m.predict(B[cs])))
        log(f"   fold {k + 1}/5 fusion={cv['fusion'][-1]:.2f} cgm_only={cv['cgm_only'][-1]:.2f}")
    M["cross_validation"] = {k: {"folds": v, "mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in cv.items()}

    # ------------------------------------------------------------------ artefacts
    log("12/12 saving models, demo data, figures")
    meds = tr[cols].median()
    for nm, obj in (("fusion_reg", regs["fusion"]), ("fusion_reg30", reg30), ("fusion_reg60", reg60), ("fusion_q10", q10),
                    ("fusion_q90", q90), ("fusion_clf", clfs["fusion"]), ("fusion_hypo", hypo_clfs["fusion"])):
        joblib.dump(obj, ROOT / "models" / f"{nm}.joblib", compress=3)
    (ROOT / "models" / "meta.json").write_text(json.dumps({
        "features": cols, "spike_threshold_prob": thr["fusion"], "hypo_threshold_prob": thr["hypo_fusion"],
        "horizon_min": HORIZON_MIN, "spike_mgdl": SPIKE_MGDL, "hypo_mgdl": HYPO_MGDL,
        "feature_medians": {k: float(v) for k, v in meds.items()},
        "groups": {k: [c for c in v if c in cols] for k, v in GROUPS.items()}}, indent=2))
    (ROOT / "results" / "metrics.json").write_text(json.dumps(M, indent=2))

    te_p = ehr[ehr.patient_id.isin(te_ids)].set_index("patient_id")
    n_hypo = sensors[sensors.patient_id.isin(te_ids) & (sensors.glucose_true < HYPO_MGDL)].groupby("patient_id").size()
    by_a1c = te_p.sort_values("hba1c")
    picks = list(by_a1c.index[np.linspace(0, len(by_a1c) - 1, 10).round().astype(int)])
    for pid in n_hypo.sort_values(ascending=False).index:
        if len(picks) >= 12:
            break
        if pid not in picks:
            picks.append(pid)
    ehr[ehr.patient_id.isin(picks)].to_csv(ROOT / "data" / "demo_ehr.csv", index=False)
    sensors[sensors.patient_id.isin(picks)].to_csv(ROOT / "data" / "demo_sensors.csv.gz", index=False)

    if make_figs:
        from .figures import make_all
        make_all(ROOT / "results" / "figures", dict(
            M=M, te=te, te_o=te_o, abs_pred=te.g_now.values + delta["fusion"], lo=lo, hi=hi,
            prob_o=prob_o, imp=imp, gimp=gimp, agp=agp, fam=fam))
    log(f"done in {time.time() - t0:.0f}s")
    return M


if __name__ == "__main__":
    main()
