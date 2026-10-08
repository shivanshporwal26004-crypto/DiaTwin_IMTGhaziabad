"""Synthetic EHR generator for Type 2 Diabetes patients (India-flavoured cohort).

Produces one row per patient with demographics, labs, comorbidities, a genetic
risk marker, medication and lifestyle. No real patient data is used anywhere in
this project (DPDP Act / HIPAA safe by construction).

For a Synthea-based EHR instead, see ``src/import_synthea.py`` which maps
Synthea CSV exports onto this same schema.
"""
import numpy as np
import pandas as pd

MEDICATIONS = ["none", "metformin", "metformin+sulfonylurea", "insulin"]
ACTIVITY = ["sedentary", "light", "moderate"]


def generate_ehr(n_patients: int = 240, seed: int = 42, age_mu: float = 52, bmi_add: float = 0.0,
                 dur_scale: float = 1.0, id_prefix: str = "P") -> pd.DataFrame:
    """age_mu / bmi_add / dur_scale shift the cohort (used for the distribution-shift test)."""
    rng = np.random.default_rng(seed)
    n = n_patients

    age = np.clip(rng.normal(age_mu, 11, n), 28, 82).round().astype(int)
    sex = rng.choice(["F", "M"], n, p=[0.47, 0.53])
    bmi = np.clip(rng.normal(27.5, 4.2, n) + bmi_add + (sex == "F") * 0.6, 18.5, 42).round(1)
    duration = np.clip(rng.gamma(2.2, 3.0 * dur_scale, n), 0.2, 25).round(1)          # years since diagnosis
    tcf7l2 = rng.choice([0, 1, 2], n, p=[0.45, 0.43, 0.12])              # TCF7L2 risk-allele count
    family_history = (rng.random(n) < (0.35 + 0.15 * (tcf7l2 > 0))).astype(int)

    hba1c = np.clip(
        6.5 + 0.045 * (bmi - 25) + 0.06 * duration + 0.20 * tcf7l2 + rng.normal(0, 0.7, n),
        5.9, 12.0,
    ).round(1)
    eag = 28.7 * hba1c - 46.7                                             # ADA eAG formula
    fasting = np.clip(0.82 * eag + rng.normal(0, 10, n), 85, 260).round().astype(int)

    egfr = np.clip(105 - 0.7 * (age - 30) - 0.8 * duration + rng.normal(0, 8, n), 25, 120).round().astype(int)
    ldl = np.clip(rng.normal(115, 28, n), 50, 220).round().astype(int)
    p_htn = np.clip(0.15 + 0.008 * (age - 30) + 0.01 * (bmi - 25), 0.05, 0.9)
    hypertension = (rng.random(n) < p_htn).astype(int)

    # medication escalates with HbA1c
    med = []
    for h in hba1c:
        if h < 6.8:
            med.append(rng.choice(["none", "metformin"], p=[0.35, 0.65]))
        elif h < 8.0:
            med.append(rng.choice(["metformin", "metformin+sulfonylurea"], p=[0.7, 0.3]))
        else:
            med.append(rng.choice(["metformin+sulfonylurea", "insulin"], p=[0.6, 0.4]))
    medication = np.array(med)

    activity = rng.choice(ACTIVITY, n, p=[0.40, 0.40, 0.20])
    vegetarian = (rng.random(n) < 0.40).astype(int)
    carb_scale = np.clip(rng.normal(1.0, 0.12, n) + 0.06 * vegetarian, 0.7, 1.3).round(2)   # meal-size multiplier
    sleep_hours_mean = np.clip(rng.normal(6.6, 0.7, n), 4.8, 8.2).round(1)

    return pd.DataFrame({
        "patient_id": [f"{id_prefix}{i:03d}" for i in range(n)],
        "age": age, "sex": sex, "bmi": bmi, "duration_years": duration,
        "hba1c": hba1c, "fasting_glucose": fasting, "egfr": egfr, "ldl": ldl,
        "hypertension": hypertension, "family_history": family_history,
        "tcf7l2_risk_alleles": tcf7l2, "medication": medication,
        "activity_level": activity, "vegetarian": vegetarian,
        "carb_scale": carb_scale, "sleep_hours_mean": sleep_hours_mean,
    })


if __name__ == "__main__":
    df = generate_ehr()
    print(df.describe(include="all").T.head(20))
