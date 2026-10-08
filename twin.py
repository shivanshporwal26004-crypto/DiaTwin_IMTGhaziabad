"""The Digital Twin: a per-patient virtual replica that ingests the two data streams,
forecasts the next 2 hours, raises alerts, explains them and simulates what-if scenarios.

Two layers:
  * Learned layer     - gradient-boosted models trained on fused EHR + sensor features
                        (30/60/120-minute forecasts, 10-90% band, spike-onset and hypoglycaemia risk).
  * Mechanistic layer - personalised meal / exercise kernels (insulin sensitivity derived
                        from the EHR) used to answer "what if I eat X / walk Y" questions.
Plus an explanation layer (feature-group neutralisation) and data-quality awareness
(CGM gaps, wearables offline, unlogged meals).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import HYPO_MGDL, SPIKE_MGDL, STEP_MIN
from .features import build_features
from .simulate_sensors import exercise_effect, insulin_sensitivity, meal_response

ROOT = Path(__file__).resolve().parent.parent
MAX_BUFFER = 400          # ~33 h of 5-minute readings


def load_bundle(models_dir=None) -> dict:
    d = Path(models_dir) if models_dir else ROOT / "models"
    meta = json.loads((d / "meta.json").read_text())
    return {
        "reg": joblib.load(d / "fusion_reg.joblib"),
        "reg30": joblib.load(d / "fusion_reg30.joblib"),
        "reg60": joblib.load(d / "fusion_reg60.joblib"),
        "q10": joblib.load(d / "fusion_q10.joblib"),
        "q90": joblib.load(d / "fusion_q90.joblib"),
        "clf": joblib.load(d / "fusion_clf.joblib"),
        "hypo": joblib.load(d / "fusion_hypo.joblib"),
        **meta,
    }


class DigitalTwin:
    def __init__(self, ehr_row: pd.Series, bundle: dict):
        self.ehr = ehr_row
        self.b = bundle
        self.buf = pd.DataFrame()

    @classmethod
    def from_history(cls, ehr_row, history: pd.DataFrame, bundle: dict):
        tw = cls(ehr_row, bundle)
        tw.ingest(history)
        return tw

    # ---- data stream 2: dynamic ------------------------------------------------
    def ingest(self, readings: pd.DataFrame):
        self.buf = pd.concat([self.buf, readings], ignore_index=True).tail(MAX_BUFFER).reset_index(drop=True)

    # ---- fusion -----------------------------------------------------------------
    def fused_row(self) -> pd.DataFrame:
        f = build_features(self.buf, self.ehr.to_frame().T.infer_objects(), with_targets=False, drop_warmup=False)
        return f.iloc[[-1]]

    def last_glucose(self) -> float:
        """Last real CGM reading (the twin keeps working through sensor gaps)."""
        return float(self.buf["glucose"].ffill().iloc[-1])

    # ---- data quality -----------------------------------------------------------
    def data_quality(self) -> dict:
        tail = self.buf.tail(6)                                    # last 30 minutes
        gap = int(self.fused_row()["cgm_gap_min"].iloc[0])
        wear = bool(tail[["steps", "hr", "hrv"]].notna().any(axis=None))
        meals24 = int(self.buf.tail(288)["meal_carbs_g"].notna().sum())
        issues = []
        if gap > 0:
            issues.append(f"CGM gap of {gap} min - forecast uses the last real reading" if gap <= 30 else f"CGM gap of {gap} min - forecast is unreliable")
        if not wear:
            issues.append("Wearables offline - forecast falls back to CGM + EHR")
        if meals24 == 0:
            issues.append("No meals logged in the last 24 h - meal effects are invisible to the twin")
        return {"cgm_gap_min": gap, "wearables_online": wear, "meals_logged_24h": meals24, "issues": issues}

    # ---- learned layer -----------------------------------------------------------
    def forecast(self) -> dict:
        row = self.fused_row()
        X = row[self.b["features"]]
        g0 = self.last_glucose()
        d30, d60, d120 = (float(self.b[k].predict(X)[0]) for k in ("reg30", "reg60", "reg"))
        pred = g0 + d120
        lo = min(g0 + float(self.b["q10"].predict(X)[0]), pred)
        hi = max(g0 + float(self.b["q90"].predict(X)[0]), pred)
        p_spike = float(self.b["clf"].predict_proba(X)[0, 1])
        p_hypo = float(self.b["hypo"].predict_proba(X)[0, 1])
        ts, tsp = self.b["spike_threshold_prob"], self.b["hypo_threshold_prob"]

        if g0 > SPIKE_MGDL:
            level, msg = "HIGH NOW", "Glucose is already above 180 mg/dL."
        elif p_spike >= ts:
            level, msg = "ALERT", "High risk of crossing 180 mg/dL within 2 hours."
        elif p_spike >= 0.5 * ts:
            level, msg = "WATCH", "Moderate risk - monitor the next readings."
        else:
            level, msg = "LOW", "No excursion expected in the next 2 hours."

        if g0 < HYPO_MGDL:
            hlevel, hmsg = "LOW NOW", "Glucose is below 70 mg/dL - treat now."
        elif p_hypo >= tsp:
            hlevel, hmsg = "ALERT", "Risk of dropping below 70 mg/dL within 2 hours."
        else:
            hlevel, hmsg = "OK", "No hypoglycaemia expected."
        return {"now": g0, "pred_30": g0 + d30, "pred_60": g0 + d60, "pred_120": pred, "lo": lo, "hi": hi,
                "spike_prob": p_spike, "threshold": ts, "level": level, "message": msg,
                "hypo_prob": p_hypo, "hypo_threshold": tsp, "hypo_level": hlevel, "hypo_message": hmsg,
                "ts": row["ts"].iloc[0]}

    # ---- explanation layer -------------------------------------------------------
    def explain(self) -> pd.DataFrame:
        """Why does the twin predict this? Neutralise each feature group to its training median and
        measure how the predicted 2 h change and the spike risk move. Positive = pushes glucose up."""
        row = self.fused_row()
        X = row[self.b["features"]]
        base_d = float(self.b["reg"].predict(X)[0])
        base_p = float(self.b["clf"].predict_proba(X)[0, 1])
        out = []
        for name, cols in self.b["groups"].items():
            Xn = X.copy()
            for c in cols:
                Xn[c] = self.b["feature_medians"][c]
            out.append({"group": name,
                        "effect_on_2h_change_mgdl": base_d - float(self.b["reg"].predict(Xn)[0]),
                        "effect_on_spike_risk_pct": (base_p - float(self.b["clf"].predict_proba(Xn)[0, 1])) * 100})
        return pd.DataFrame(out).set_index("group")

    # ---- mechanistic what-if ---------------------------------------------------
    def what_if(self, planned_carbs=70.0, meal_in_min=15, walk_min=20, horizon_min=180) -> dict:
        isf = insulin_sensitivity(self.ehr)
        n = horizon_min // STEP_MIN + 1
        tau = np.arange(n) * STEP_MIN
        g0 = self.last_glucose()

        pending = np.zeros(n)                                       # response still to come from meals logged in the last 4 h
        tail = self.buf.tail(48).reset_index(drop=True)
        for i, c in tail["meal_carbs_g"].items():
            if pd.notna(c) and c > 0:
                age = (len(tail) - 1 - i) * STEP_MIN
                pending += meal_response(age + tau, c, isf) - meal_response(age, c, isf)

        def walk(start_min, minutes):
            s = np.zeros(n)
            i0, i1 = int(start_min // STEP_MIN), int((start_min + minutes) // STEP_MIN)
            s[i0:i1] = 500
            return exercise_effect(s)

        eat = pending + meal_response(tau - meal_in_min, planned_carbs, isf)
        half = pending + meal_response(tau - meal_in_min, planned_carbs * 0.5, isf)
        scen = {
            f"Eat {planned_carbs:.0f} g carbs": eat,
            f"Eat half ({planned_carbs * 0.5:.0f} g)": half,
            f"Eat + {walk_min} min walk after": eat - walk(meal_in_min + 10, walk_min),
            "Skip the meal": pending,
        }
        curves = {k: np.clip(g0 + v, 40, 400) for k, v in scen.items()}
        return {"minutes": tau, "curves": curves, "insulin_sensitivity": isf}

    # ---- clinician-facing summary ---------------------------------------------
    def summary(self) -> list:
        f = self.forecast()
        r = self.fused_row().iloc[0]
        day = self.buf.tail(288)
        gl = day["glucose"].dropna()
        tir = float(((gl >= HYPO_MGDL) & (gl <= SPIKE_MGDL)).mean() * 100)
        e = self.ehr
        notes = [
            f"Glucose now {f['now']:.0f} mg/dL; forecast {f['pred_30']:.0f} / {f['pred_60']:.0f} / {f['pred_120']:.0f} mg/dL "
            f"at +30 / +60 / +120 min (80% band at 2 h: {f['lo']:.0f}-{f['hi']:.0f}). Spike-onset risk {f['spike_prob'] * 100:.0f}%.",
            f"Time in range (70-180) over the last 24 h: {tir:.0f}%.",
            f"Profile: HbA1c {e['hba1c']:.1f}%, fasting glucose {e['fasting_glucose']:.0f} mg/dL, BMI {e['bmi']:.1f}, on {e['medication']}.",
        ]
        if f["hypo_level"] != "OK":
            notes.append(f"Hypoglycaemia watch: {f['hypo_message']} Review dosing and carbohydrate timing.")
        if pd.notna(r["sleep_h_24h"]) and r["sleep_h_24h"] < 6:
            notes.append(f"Sleep debt: only {r['sleep_h_24h']:.1f} h in the last 24 h - expect higher fasting glucose.")
        if pd.notna(r["steps_1h"]) and r["steps_1h"] < 300 and r["min_since_meal"] < 120:
            notes.append("Little movement since the last meal - a short walk would likely blunt the peak.")
        if e["hba1c"] >= 8.0:
            notes.append("HbA1c >= 8%: consider reviewing therapy at the next visit.")
        notes += [f"Data quality: {i}." for i in self.data_quality()["issues"]]
        return notes
