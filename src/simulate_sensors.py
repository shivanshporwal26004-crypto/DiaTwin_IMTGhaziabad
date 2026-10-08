"""Synthetic wearable / CGM time-series conditioned on each patient's EHR.

Streams (5-minute resolution): glucose (CGM, with sensor dropouts), heart rate, HRV (RMSSD),
steps, sleep stage and a logged-carbohydrate meal diary. ``glucose_true`` is the noise-free
physiological truth used only for evaluation targets.

Physiology (deliberately simple, transparent and documented in the README):
  glucose = basal(EHR, dawn effect, sleep debt, stress)
          + sum(meal responses)            # gamma-like kernel, peak ~55 min
          - drug effect                    # bolus insulin / sulfonylurea dosed for the planned meal
                                           #   (skipped or smaller meals => hypoglycaemia risk)
          - exercise effect                # steps lower glucose for ~90 min
          + AR(1) physiological noise + CGM sensor noise
Insulin sensitivity is derived from HbA1c, BMI, age, disease duration,
TCF7L2 risk alleles, medication and activity level, so the *same* meal produces
a different curve in different patients - this is what the twin must learn.

The same kernels are reused by ``twin.py`` for what-if scenarios.
"""
import numpy as np
import pandas as pd

from .config import STEP_MIN, STEPS_PER_DAY

MED_ISF = {"none": 1.0, "metformin": 1.15, "metformin+sulfonylurea": 1.30, "insulin": 1.35}
ACT_ISF = {"sedentary": 0.90, "light": 1.0, "moderate": 1.12}
ACT_STEPS = {"sedentary": 3500, "light": 6000, "moderate": 9000}
ACT_WALK_P = {"sedentary": 0.20, "light": 0.40, "moderate": 0.70}
GAIN = 1.25          # mg/dL rise per gram of carbohydrate at average insulin sensitivity
PEAK_MIN = 55.0      # minutes to post-meal peak


def insulin_sensitivity(p) -> float:
    """Relative insulin sensitivity (1.0 = average T2D patient). Higher = flatter meal response."""
    s = np.exp(-0.18 * (p["hba1c"] - 7.5))
    s *= np.exp(-0.035 * (p["bmi"] - 27))
    s *= np.exp(-0.008 * (p["age"] - 52))
    s *= np.exp(-0.02 * (p["duration_years"] - 5))
    s *= np.exp(-0.10 * p["tcf7l2_risk_alleles"])
    s *= MED_ISF[p["medication"]] * ACT_ISF[p["activity_level"]]
    return float(s)


def meal_response(minutes_since, carbs, isf, gi=1.0, peak=PEAK_MIN):
    """Glucose rise (mg/dL) `minutes_since` minutes after a meal of `carbs` grams."""
    tau = np.asarray(minutes_since, dtype=float)
    shape = np.where(tau > 0, (tau / peak) ** 2 * np.exp(2 * (1 - tau / peak)), 0.0)
    return GAIN * carbs * gi / isf * shape


def exercise_effect(steps) -> np.ndarray:
    """Glucose lowering (mg/dL) from a steps-per-interval series (decays over ~90 min)."""
    steps = np.asarray(steps, dtype=float)
    kernel = np.exp(-np.arange(18) / 10.0)
    return 0.013 * np.convolve(steps, kernel)[: len(steps)]


def _sleep_stages(n, days, bed, sleep_h, rng):
    stage = np.zeros(n, dtype=int)
    wake0 = int(max(0.0, bed[0] + sleep_h[0] - 24) * 12)
    ranges = [(0, wake0)]
    for k in range(days):
        s = k * STEPS_PER_DAY + int(bed[k] * 12)
        e = s + int(sleep_h[min(k + 1, days - 1)] * 12)
        ranges.append((s, min(e, n)))
    for s, e in ranges:
        if e <= s:
            continue
        blocks = rng.choice([1, 2, 3], size=(e - s) // 6 + 1, p=[0.55, 0.25, 0.20])
        stage[s:e] = np.repeat(blocks, 6)[: e - s]
    return stage


def simulate_patient(p, days: int, rng: np.random.Generator) -> pd.DataFrame:
    n = days * STEPS_PER_DAY
    t = np.arange(n)
    hour = (t * STEP_MIN / 60.0) % 24
    day = t // STEPS_PER_DAY
    isf = insulin_sensitivity(p)

    stress = np.clip(rng.beta(2, 5, days), 0, 1)
    sleep_h = np.clip(rng.normal(p["sleep_hours_mean"], 0.9, days), 3.5, 9.0)
    bed = np.clip(rng.normal(23.0, 0.5, days), 21.5, 24.0)
    stage = _sleep_stages(n, days, bed, sleep_h, rng)
    asleep = stage > 0

    # ---- meals -----------------------------------------------------------
    meal_true = np.zeros(n)
    meal_logged = np.full(n, np.nan)
    meal_curve = np.zeros(n)
    meal_idx_list = []
    drug_curve = np.zeros(n)                               # glucose-lowering from sulfonylurea / bolus insulin
    plan = [(8.0, 0.6, 55, 12, 0.95), (13.3, 0.7, 78, 15, 1.0), (17.0, 0.8, 28, 8, 0.6), (20.5, 0.7, 72, 14, 1.0)]
    insulin = p["medication"] == "insulin"
    sulf = "sulfonylurea" in p["medication"]
    for d in range(days):
        for h0, hs, c_mu, c_sd, prob in plan:
            idx = d * STEPS_PER_DAY + int(np.clip(rng.normal(h0, hs), 5, 23.5) * 12)
            if idx >= n - 1:
                continue
            tau = (np.arange(n - idx)) * STEP_MIN
            eaten = rng.random() < prob * 0.96                     # ~4% extra skipped meals
            nominal = c_mu * p["carb_scale"]
            if eaten:
                carbs = max(8.0, rng.normal(c_mu, c_sd) * p["carb_scale"] * float(np.exp(rng.normal(0, 0.12))))
                gi = float(np.exp(rng.normal(0, 0.15)))
                meal_curve[idx:] += meal_response(tau, carbs, isf, gi)[: n - idx]
                meal_true[idx] += carbs
                if rng.random() < 0.85:                    # diary logging is imperfect
                    meal_logged[idx] = round(carbs)
                meal_idx_list.append(idx)
            # medication is dosed for the *planned* meal; a skipped meal can still be dosed
            if (insulin or sulf) and (eaten or rng.random() < 0.5):
                k, peak = (1.35, 45.0) if insulin else (0.55, 100.0)
                mismatch = float(np.exp(rng.normal(0, 0.25 if insulin else 0.15)))
                drug_curve[idx:] += k * mismatch * meal_response(tau, nominal, isf, 1.0, peak)[: n - idx]

    # ---- steps -----------------------------------------------------------
    waking_rate = ACT_STEPS[p["activity_level"]] * 0.55 / (16 * 12)
    profile = np.where((hour > 6) & (hour < 22), 1.0, 0.0) * (0.6 + 0.8 * np.exp(-((hour - 17) ** 2) / 18))
    steps = rng.poisson(waking_rate * profile).astype(float)
    steps[asleep] = 0
    for idx in meal_idx_list:
        if rng.random() < ACT_WALK_P[p["activity_level"]] and 10 <= hour[idx] <= 22:
            s = idx + int(rng.integers(1, 4))
            dur = int(rng.integers(2, 7))
            steps[s: s + dur] += rng.integers(350, 650, size=len(steps[s: s + dur]))
    if p["activity_level"] == "moderate":
        for d in range(days):
            if rng.random() < 0.5:
                s = d * STEPS_PER_DAY + 6 * 12 + 6
                steps[s: s + 6] += rng.integers(400, 700, size=6)
    steps = np.where(asleep, 0, steps)

    # ---- glucose ---------------------------------------------------------
    sleep_prev = np.concatenate([[sleep_h[0]], sleep_h[:-1]])
    basal = (
        p["fasting_glucose"]
        + 12 * np.exp(-((hour - 6.5) ** 2) / (2 * 1.3 ** 2))
        + 5 * np.maximum(0, 7 - sleep_prev)[day]
        + 9 * stress[day]
    )
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.95 * eps[i - 1] + rng.normal(0, 1.6)
    glucose_true = basal + meal_curve - drug_curve - exercise_effect(steps) + eps + rng.normal(0, 2.0, n)
    glucose_true = np.clip(glucose_true, 40, 400)
    glucose = glucose_true.copy()                          # what the CGM reports (with dropouts)
    for _ in range(rng.poisson(0.7 * days)):
        s0 = int(rng.integers(0, n - 12))
        glucose[s0: s0 + int(rng.integers(3, 11))] = np.nan

    # ---- heart rate / HRV -----------------------------------------------
    spm = steps / STEP_MIN
    rest_hr = 62 + 0.15 * (p["bmi"] - 25) + 5 * stress[day]
    hr = rest_hr + 0.35 * spm - 8 * asleep + rng.normal(0, 2.0, n)
    hrv = (
        38 - 0.25 * (p["age"] - 50) - 0.6 * (p["bmi"] - 27) - 5 * stress[day]
        + 8 * asleep - 0.05 * spm + rng.normal(0, 4.0, n)
    )
    hrv = np.clip(hrv, 8, None)

    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(t * STEP_MIN, unit="m")
    return pd.DataFrame({
        "patient_id": p["patient_id"], "ts": ts, "glucose": np.round(glucose, 1), "glucose_true": glucose_true.round(1),
        "hr": hr.round(1), "hrv": hrv.round(1), "steps": steps.astype(int),
        "sleep_stage": stage, "meal_carbs_g": meal_logged,
    })


def simulate_cohort(ehr: pd.DataFrame, days: int = 14, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_patient(row, days, rng) for _, row in ehr.iterrows()]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    from .synth_ehr import generate_ehr
    e = generate_ehr(5)
    s = simulate_cohort(e, days=3)
    print(s.describe().T)
