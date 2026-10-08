"""Evaluation toolkit: patient-level bootstrap CIs, calibration, alert burden and lead time,
subgroup analysis, robustness perturbations and standard CGM glycaemic metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import HYPO_MGDL, SPIKE_MGDL, STEP_MIN


# ---------------------------------------------------------------- glycaemic metrics
def glycemic_metrics(sensors: pd.DataFrame) -> dict:
    """International-consensus CGM metrics (time in range, GMI, CV) on the physiological truth."""
    g = sensors["glucose_true"]
    per = sensors.groupby("patient_id")["glucose_true"]
    cv = (per.std() / per.mean() * 100).mean()
    return {
        "mean_glucose_mgdl": float(g.mean()),
        "gmi_pct": float(3.31 + 0.02392 * g.mean()),
        "cv_pct": float(cv),
        "tir_70_180_pct": float(((g >= HYPO_MGDL) & (g <= SPIKE_MGDL)).mean() * 100),
        "tar_over_180_pct": float((g > SPIKE_MGDL).mean() * 100),
        "tar_over_250_pct": float((g > 250).mean() * 100),
        "tbr_under_70_pct": float((g < HYPO_MGDL).mean() * 100),
        "tbr_under_54_pct": float((g < 54).mean() * 100),
        "cgm_dropout_pct": float(sensors["glucose"].isna().mean() * 100),
    }


def agp_profile(sensors: pd.DataFrame) -> pd.DataFrame:
    """Ambulatory Glucose Profile: percentile bands of glucose by time of day."""
    t = sensors["ts"].dt.hour * 60 + sensors["ts"].dt.minute
    q = sensors.groupby(t)["glucose_true"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    q.columns = ["p05", "p25", "p50", "p75", "p95"]
    return q


# ---------------------------------------------------------------- bootstrap
def patient_bootstrap_mae(df, pred_delta_by_model: dict, n_boot=1000, seed=0):
    """Patient-level bootstrap of MAE for each model and the paired difference vs the first one."""
    rng = np.random.default_rng(seed)
    pid = df["patient_id"].values
    uniq, inv = np.unique(pid, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    sums = {}
    for name, pd_ in pred_delta_by_model.items():
        err = np.abs(df["g_now"].values + pd_ - df["y_g120"].values) if pd_ is not None else np.abs(df["g_now"].values - df["y_g120"].values)
        sums[name] = np.bincount(inv, weights=err)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    out = {}
    for name, sm in sums.items():
        boot = sm[idx].sum(1) / cnt[idx].sum(1)
        out[name] = {"mae": float(sm.sum() / cnt.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    names = list(sums)
    for other in names[1:]:
        diff = (sums[names[0]][idx].sum(1) - sums[other][idx].sum(1)) / cnt[idx].sum(1)     # baseline - model
        out[other]["gain_vs_" + names[0]] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return out


def patient_bootstrap_auc(df, prob, label="y_spike", n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    pid = df["patient_id"].values
    uniq = np.unique(pid)
    groups = {u: np.where(pid == u)[0] for u in uniq}
    y = df[label].values
    aucs = []
    for _ in range(n_boot):
        ids = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([groups[i] for i in ids])
        if y[ix].min() == y[ix].max():
            continue
        aucs.append(roc_auc_score(y[ix], prob[ix]))
    return {"auc": float(roc_auc_score(y, prob)), "ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}


def paired_auc_gain(df, prob_a, prob_b, label="y_spike", n_boot=200, seed=0):
    """AUC(b) - AUC(a) with patient-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    pid = df["patient_id"].values
    uniq = np.unique(pid)
    groups = {u: np.where(pid == u)[0] for u in uniq}
    y = df[label].values
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[i] for i in rng.choice(uniq, len(uniq), replace=True)])
        if y[ix].min() == y[ix].max():
            continue
        d.append(roc_auc_score(y[ix], prob_b[ix]) - roc_auc_score(y[ix], prob_a[ix]))
    return {"mean": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


# ---------------------------------------------------------------- calibration
def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ix = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = ix == b
        if m.sum() >= 20:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    ece = sum(n * abs(a - b) for a, b, n in rows) / sum(n for _, _, n in rows)
    return {"curve": rows, "brier": float(brier_score_loss(y, p)), "ece": float(ece)}


# ---------------------------------------------------------------- alert burden / lead time
def alert_metrics(te: pd.DataFrame, prob: np.ndarray, thr: float, rows_per_step=3) -> dict:
    """Operational view of the spike alert on contiguous 15-minute rows.

    alerts/patient-day  : number of distinct alert episodes per patient per day
    false-alert share   : episodes not followed by a >180 crossing within 2 h
    detected / lead     : share of true excursions (first crossing of 180) preceded by an alert, and by how long
    """
    d = te[["patient_id", "step_idx", "g_now"]].copy()
    d["alert"] = (prob >= thr) & (d["g_now"].values <= SPIKE_MGDL)
    d = d.sort_values(["patient_id", "step_idx"])
    look = 8                                        # 8 rows x 15 min = 120 min
    episodes = false_eps = events = det = det30 = 0
    leads = []
    n_rows = 0
    for _, g in d.groupby("patient_id"):
        a, v, st = g["alert"].values, g["g_now"].values, g["step_idx"].values
        n_rows += len(g)
        contiguous = np.r_[False, np.diff(st) == rows_per_step]
        start = a & ~np.r_[False, a[:-1]]
        cross = np.r_[False, (v[1:] > SPIKE_MGDL) & (v[:-1] <= SPIKE_MGDL)] & contiguous
        for k in np.where(start)[0]:
            episodes += 1
            if not (v[k + 1: k + 1 + look] > SPIKE_MGDL).any():
                false_eps += 1
        for i in np.where(cross)[0]:
            events += 1
            w = a[max(0, i - look): i]
            if w.any():
                lead = (len(w) - int(np.argmax(w))) * STEP_MIN * rows_per_step
                det += 1
                det30 += lead >= 30
                leads.append(lead)
    days = n_rows * STEP_MIN * rows_per_step / 1440
    return {
        "alert_episodes_per_patient_day": episodes / max(days, 1e-9),
        "false_alert_share_pct": 100 * false_eps / max(episodes, 1),
        "excursions": int(events),
        "detected_pct": 100 * det / max(events, 1),
        "detected_with_30min_lead_pct": 100 * det30 / max(events, 1),
        "median_lead_min": float(np.median(leads)) if leads else 0.0,
        "lead_times_min": [float(x) for x in leads],
    }


# ---------------------------------------------------------------- subgroup analysis
def subgroup_table(te: pd.DataFrame, pred: dict, te_o: pd.DataFrame, prob_o: np.ndarray) -> list:
    """MAE (persistence / CGM-only / fusion) and spike ROC-AUC per patient subgroup."""
    def groups(d):
        med = np.where(d.med_insulin == 1, "insulin", np.where(d.med_sulfonylurea == 1, "sulfonylurea", "metformin / none"))
        return {
            "HbA1c": pd.cut(d.hba1c, [0, 7, 8, 20], labels=["<7%", "7-8%", ">=8%"], right=False).astype(str),
            "Medication": pd.Series(med, index=d.index),
            "Age": pd.cut(d.age, [0, 50, 65, 120], labels=["<50", "50-64", ">=65"], right=False).astype(str),
            "Sex": d.sex_male.map({0: "female", 1: "male"}),
            "Activity": d.activity_code.map({0: "sedentary", 1: "light", 2: "moderate"}),
        }
    g_te, g_o = groups(te), groups(te_o)
    rows = []
    for dim in g_te:
        for lvl in sorted(g_te[dim].unique()):
            m = (g_te[dim] == lvl).values
            r = {"dimension": dim, "group": lvl, "rows": int(m.sum()),
                 "patients": int(te.loc[m, "patient_id"].nunique()),
                 "mae_persistence": float(np.abs(te.g_now.values[m] - te.y_g120.values[m]).mean())}
            for k, dl in pred.items():
                r[f"mae_{k}"] = float(np.abs(te.g_now.values[m] + dl[m] - te.y_g120.values[m]).mean())
            mo = (g_o[dim] == lvl).values
            yo = te_o.y_spike.values[mo]
            r["spike_auc_fusion"] = float(roc_auc_score(yo, prob_o[mo])) if mo.sum() > 50 and yo.min() != yo.max() else None
            rows.append(r)
    return rows


# ---------------------------------------------------------------- robustness perturbations
def perturb(sensors: pd.DataFrame, kind: str, seed=0) -> pd.DataFrame:
    """Return a degraded copy of the *observed* streams. Targets use glucose_true so are unaffected."""
    rng = np.random.default_rng(seed)
    s = sensors.copy()
    if kind == "clean":
        return s
    if kind == "cgm_noise_8mgdl":
        s["glucose"] = s["glucose"] + rng.normal(0, 8, len(s))
    elif kind == "cgm_dropout_20pct":
        n = len(s)
        for _ in range(int(n * 0.20 / 8)):               # ~20% of readings lost, in ~40-minute gaps
            a = int(rng.integers(0, n - 8))
            s.iloc[a: a + 8, s.columns.get_loc("glucose")] = np.nan
    elif kind == "no_meal_log":
        s["meal_carbs_g"] = np.nan
    elif kind == "wearables_offline":
        for c in ("steps", "hr", "hrv"):
            s[c] = np.nan
        s["sleep_stage"] = np.nan
    else:
        raise ValueError(kind)
    return s
