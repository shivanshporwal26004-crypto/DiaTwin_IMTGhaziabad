"""Optional: map Synthea CSV output onto the DiaTwin EHR schema.

Usage:
    java -jar synthea-with-dependencies.jar -p 300 --exporter.csv.export=true
    python -m src.import_synthea path/to/synthea/output/csv data/ehr_synthea.csv

Only patients with a diabetes condition are kept. Lab values come from the
observations table (LOINC 4548-4 HbA1c, 39156-5 BMI, 2339-0 glucose,
33914-3 eGFR, 18262-6 LDL). Fields Synthea does not model (genetic marker,
lifestyle) are drawn from the same distributions as ``synth_ehr.py``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOINC = {"hba1c": "4548-4", "bmi": "39156-5", "glucose": "2339-0", "egfr": "33914-3", "ldl": "18262-6"}


def _latest(obs: pd.DataFrame, code: str) -> pd.Series:
    sub = obs[obs["CODE"] == code].sort_values("DATE")
    return pd.to_numeric(sub.groupby("PATIENT")["VALUE"].last(), errors="coerce")


def import_synthea(csv_dir: str, seed: int = 0) -> pd.DataFrame:
    d = Path(csv_dir)
    pts = pd.read_csv(d / "patients.csv")
    cond = pd.read_csv(d / "conditions.csv")
    obs = pd.read_csv(d / "observations.csv")

    desc = cond["DESCRIPTION"].fillna("")
    is_dm = desc.str.contains("diabetes", case=False) & ~desc.str.contains("prediabetes", case=False)   # exclude prediabetes
    diabetic = cond[is_dm]
    start = diabetic.groupby("PATIENT")["START"].min()
    pts = pts[pts["Id"].isin(start.index)].set_index("Id")

    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=pts.index)
    ref = pd.Timestamp.today()
    out["age"] = ((ref - pd.to_datetime(pts["BIRTHDATE"])).dt.days / 365.25).round().astype(int)
    out["sex"] = pts["GENDER"].map({"F": "F", "M": "M"}).fillna("F")
    out["duration_years"] = ((ref - pd.to_datetime(start.reindex(out.index))).dt.days / 365.25).round(1)
    for name, code in LOINC.items():
        out[name] = _latest(obs, code).reindex(out.index)
    out["hba1c"] = out["hba1c"].fillna(out["hba1c"].median() if out["hba1c"].notna().any() else 7.5)
    out["bmi"] = out["bmi"].fillna(out["bmi"].median() if out["bmi"].notna().any() else 27.0)
    out["fasting_glucose"] = out["glucose"].fillna(0.75 * (28.7 * out["hba1c"] - 46.7)).round().astype(int)
    out["egfr"] = out["egfr"].fillna(90).round().astype(int)
    out["ldl"] = out["ldl"].fillna(115).round().astype(int)
    out = out.drop(columns=["glucose"])

    n = len(out)
    has_htn = cond[cond["DESCRIPTION"].str.contains("hypertension", case=False, na=False)]["PATIENT"].unique()
    out["hypertension"] = out.index.isin(has_htn).astype(int)
    out["family_history"] = (rng.random(n) < 0.4).astype(int)
    out["tcf7l2_risk_alleles"] = rng.choice([0, 1, 2], n, p=[0.45, 0.43, 0.12])
    out["medication"] = np.where(out["hba1c"] < 8, "metformin", "metformin+sulfonylurea")
    out["activity_level"] = rng.choice(["sedentary", "light", "moderate"], n, p=[0.4, 0.4, 0.2])
    out["vegetarian"] = (rng.random(n) < 0.4).astype(int)
    out["carb_scale"] = np.clip(rng.normal(1.0, 0.12, n), 0.7, 1.3).round(2)
    out["sleep_hours_mean"] = np.clip(rng.normal(6.6, 0.7, n), 4.8, 8.2).round(1)
    out = out.reset_index(drop=True)
    out.insert(0, "patient_id", [f"S{i:03d}" for i in range(n)])
    return out


if __name__ == "__main__":
    df = import_synthea(sys.argv[1])
    df.to_csv(sys.argv[2], index=False)
    print(f"Wrote {len(df)} Synthea diabetic patients to {sys.argv[2]}")
