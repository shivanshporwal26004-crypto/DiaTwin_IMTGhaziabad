"""Feature engineering: fuses the dynamic sensor stream with the static EHR profile.

Four feature groups (used for the ablation study):
    CGM       - glucose history, trends, rolling stats, time of day
    SENSOR    - wearables (steps, HR, HRV, sleep) + logged meals
    EHR       - static profile (labs, comorbidities, genetics, medication, lifestyle)
Feature sets:
    cgm_only | cgm+wearables | cgm+ehr | fusion (all three)
"""
import numpy as np
import pandas as pd

from .config import CGM_FFILL_LIMIT, HORIZON_STEPS, HYPO_MGDL, SPIKE_MGDL, STEP_MIN, WARMUP_STEPS

CGM = ["g_now", "g_lag5", "g_lag15", "g_lag30", "g_lag60", "g_lag120",
       "g_d15", "g_d30", "g_d60", "g_mean1h", "g_std1h", "g_max3h", "g_min3h",
       "g_mean24h", "cgm_gap_min", "hour_sin", "hour_cos"]
SENSOR = ["steps_15m", "steps_1h", "steps_3h", "hr_now", "hrv_1h", "sleep_h_24h",
          "hrv_sleep_24h", "carbs_1h", "carbs_3h", "min_since_meal"]
EHR = ["age", "sex_male", "bmi", "duration_years", "hba1c", "fasting_glucose", "egfr", "ldl",
       "hypertension", "family_history", "tcf7l2_risk_alleles", "med_metformin", "med_sulfonylurea",
       "med_insulin", "activity_code", "vegetarian"]

FEATURE_SETS = {
    "cgm_only": CGM,
    "cgm+wearables": CGM + SENSOR,
    "cgm+ehr": CGM + EHR,
    "fusion": CGM + SENSOR + EHR,
}

# Cold start: a newly onboarded patient with < 1 h of CGM history and no sleep history yet
COLD_CGM = ["g_now", "g_lag5", "g_lag15", "g_lag30", "g_d15", "g_d30", "cgm_gap_min", "hour_sin", "hour_cos"]
COLD_SENSOR = ["steps_15m", "steps_1h", "hr_now", "hrv_1h", "carbs_1h", "carbs_3h", "min_since_meal"]
COLD_SETS = {
    "cold_cgm": COLD_CGM,
    "cold_cgm+ehr": COLD_CGM + EHR,
    "cold_cgm+wearables": COLD_CGM + COLD_SENSOR,
    "cold_fusion": COLD_CGM + COLD_SENSOR + EHR,
}

_ACT = {"sedentary": 0, "light": 1, "moderate": 2}


def static_features(e) -> dict:
    """Encode one EHR row (Series or dict) into numeric static features."""
    med = e["medication"]
    return {
        "age": e["age"], "sex_male": int(e["sex"] == "M"), "bmi": e["bmi"],
        "duration_years": e["duration_years"], "hba1c": e["hba1c"],
        "fasting_glucose": e["fasting_glucose"], "egfr": e["egfr"], "ldl": e["ldl"],
        "hypertension": e["hypertension"], "family_history": e["family_history"],
        "tcf7l2_risk_alleles": e["tcf7l2_risk_alleles"],
        "med_metformin": int("metformin" in med), "med_sulfonylurea": int("sulfonylurea" in med),
        "med_insulin": int(med == "insulin"),
        "activity_code": _ACT[e["activity_level"]], "vegetarian": e["vegetarian"],
    }


def _patient_features(d: pd.DataFrame, e, with_targets: bool) -> pd.DataFrame:
    d = d.reset_index(drop=True)
    real = d["glucose"].notna()
    g = d["glucose"].ffill(limit=CGM_FFILL_LIMIT)          # bridge short CGM dropouts (<= 30 min)
    f = pd.DataFrame(index=d.index)
    _pos = pd.Series(np.where(real, np.arange(len(d)), np.nan)).ffill()
    f["cgm_gap_min"] = (np.arange(len(d)) - _pos) * STEP_MIN
    f["g_now"] = g
    for k in (1, 3, 6, 12, 24):
        f[f"g_lag{k * STEP_MIN}"] = g.shift(k)
    f["g_d15"] = g - g.shift(3)
    f["g_d30"] = g - g.shift(6)
    f["g_d60"] = g - g.shift(12)
    f["g_mean1h"] = g.rolling(12, min_periods=6).mean()
    f["g_std1h"] = g.rolling(12, min_periods=6).std()
    f["g_max3h"] = g.rolling(36, min_periods=12).max()
    f["g_min3h"] = g.rolling(36, min_periods=12).min()
    f["g_mean24h"] = g.rolling(288, min_periods=144).mean()
    hour = d["ts"].dt.hour + d["ts"].dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    f["steps_15m"] = d["steps"].rolling(3, min_periods=1).sum()
    f["steps_1h"] = d["steps"].rolling(12, min_periods=1).sum()
    f["steps_3h"] = d["steps"].rolling(36, min_periods=1).sum()
    f["hr_now"] = d["hr"].rolling(3, min_periods=1).mean()
    f["hrv_1h"] = d["hrv"].rolling(12, min_periods=1).mean()
    asleep = (d["sleep_stage"] > 0).where(d["sleep_stage"].notna())           # NaN stays NaN (device offline)
    f["sleep_h_24h"] = asleep.rolling(288, min_periods=144).sum() * STEP_MIN / 60
    f["hrv_sleep_24h"] = d["hrv"].where(asleep.fillna(False).astype(bool)).rolling(288, min_periods=1).mean()
    c = d["meal_carbs_g"].fillna(0)
    f["carbs_1h"] = c.rolling(12, min_periods=1).sum()
    f["carbs_3h"] = c.rolling(36, min_periods=1).sum()
    idx = np.arange(len(d))
    last = pd.Series(np.where(c > 0, idx, np.nan)).ffill()
    f["min_since_meal"] = ((idx - last) * STEP_MIN).clip(upper=480).fillna(480)

    for k, v in static_features(e).items():
        f[k] = v

    f["patient_id"] = d["patient_id"]
    f["ts"] = d["ts"]
    f["step_idx"] = idx
    if with_targets:
        gt = d["glucose_true"] if "glucose_true" in d else d["glucose"]     # physiological truth
        for h in (6, 12, HORIZON_STEPS):
            f[f"y_g{h * STEP_MIN}"] = gt.shift(-h)
        nxt = gt.shift(-1)[::-1]
        fwd_max = nxt.rolling(HORIZON_STEPS).max()[::-1]
        fwd_min = nxt.rolling(HORIZON_STEPS).min()[::-1]
        f["y_spike"] = (fwd_max > SPIKE_MGDL).astype(float).where(fwd_max.notna())
        f["y_hypo"] = (fwd_min < HYPO_MGDL).astype(float).where(fwd_min.notna())
    return f


def build_features(sensors: pd.DataFrame, ehr: pd.DataFrame, with_targets: bool = True,
                   drop_warmup: bool = True) -> pd.DataFrame:
    ehr_i = ehr.set_index("patient_id")
    frames = []
    for pid, d in sensors.groupby("patient_id", sort=False):
        frames.append(_patient_features(d, ehr_i.loc[pid], with_targets))
    out = pd.concat(frames, ignore_index=True)
    if drop_warmup:
        out = out[out["step_idx"] >= WARMUP_STEPS]
    if with_targets:
        out = out.dropna(subset=["y_g30", "y_g60", "y_g120", "y_spike", "y_hypo", "g_now"])   # g_now NaN = CGM gap > 30 min
    return out.reset_index(drop=True)
