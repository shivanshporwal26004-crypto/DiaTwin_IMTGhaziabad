import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src.config import SPIKE_MGDL, WARMUP_STEPS
from src.features import COLD_SETS, FEATURE_SETS, build_features
from src.simulate_sensors import exercise_effect, insulin_sensitivity, meal_response, simulate_cohort
from src.synth_ehr import generate_ehr
from src.twin import DigitalTwin, load_bundle


def _small(n=8, days=4):
    ehr = generate_ehr(n, seed=1)
    return ehr, simulate_cohort(ehr, days=days, seed=2)


# ------------------------------------------------------------------ data layer
def test_ehr_schema_and_ranges():
    ehr = generate_ehr(50, seed=0)
    assert ehr.patient_id.is_unique
    assert ehr.hba1c.between(5.9, 12).all() and ehr.bmi.between(18, 43).all()


def test_cohort_shift_parameters_move_the_distribution():
    base, shifted = generate_ehr(300, seed=1), generate_ehr(300, seed=1, age_mu=63, bmi_add=2.5, dur_scale=1.5)
    assert shifted.age.mean() > base.age.mean() + 5 and shifted.bmi.mean() > base.bmi.mean() + 1.5


def test_sensor_streams_shape_range_and_dropouts():
    ehr, s = _small()
    assert len(s) == 8 * 4 * 288
    assert s.glucose_true.between(40, 400).all()
    assert s.glucose.isna().any() and s.glucose_true.notna().all()          # dropouts only in the observed stream
    assert s.meal_carbs_g.notna().any() and set(s.sleep_stage.unique()) <= {0, 1, 2, 3}


def test_physiology_is_monotone_in_the_right_direction():
    p_lo = dict(hba1c=6.0, bmi=24, age=45, duration_years=2, tcf7l2_risk_alleles=0, medication="metformin", activity_level="moderate")
    p_hi = dict(p_lo, hba1c=10.0, bmi=34, duration_years=15, tcf7l2_risk_alleles=2, activity_level="sedentary")
    assert insulin_sensitivity(p_lo) > insulin_sensitivity(p_hi)
    assert meal_response(55, 70, 0.6) > meal_response(55, 70, 1.2)
    assert exercise_effect(np.r_[np.zeros(3), np.full(4, 500.0)]).max() > 15


def test_insulin_users_have_more_hypoglycaemia_than_non_users():
    ehr = generate_ehr(300, seed=5)
    s = simulate_cohort(ehr, days=10, seed=6)
    med = s.patient_id.map(ehr.set_index("patient_id").medication)
    tbr = (s.glucose_true < 70).groupby(med).mean()
    assert tbr["insulin"] > tbr.drop("insulin").max()


# ------------------------------------------------------------------ features
def test_features_have_no_target_columns_and_correct_horizons():
    ehr, s = _small()
    f = build_features(s, ehr)
    assert f.step_idx.min() >= WARMUP_STEPS
    cols = sorted({c for v in list(FEATURE_SETS.values()) + list(COLD_SETS.values()) for c in v})
    assert not any(c.startswith("y_") for c in cols)
    assert len(FEATURE_SETS["fusion"]) == 43
    assert set(f.y_spike.unique()) <= {0.0, 1.0} and set(f.y_hypo.unique()) <= {0.0, 1.0}
    assert not f[["g_now", "y_g30", "y_g60", "y_g120"]].isna().any().any()


def test_target_is_future_truth_not_observed_noise():
    ehr, s = _small(2, 3)
    f = build_features(s, ehr)
    r = f.iloc[100]
    pid = r.patient_id
    truth = s[s.patient_id == pid].reset_index(drop=True)
    assert r.y_g120 == truth.glucose_true.iloc[int(r.step_idx) + 24]


def test_missing_wearables_stay_missing_not_zero():
    ehr, s = _small(3, 3)
    f = build_features(ev.perturb(s, "wearables_offline"), ehr)
    assert f.sleep_h_24h.isna().all() and f.steps_1h.isna().all()


# ------------------------------------------------------------------ evaluation toolkit
def test_calibration_and_bootstrap_are_sane():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)                # perfectly calibrated by construction
    c = ev.calibration(y, p)
    assert c["ece"] < 0.05 and 0.15 < c["brier"] < 0.2


def test_alert_metrics_detects_an_obvious_lead_time():
    # one patient: glucose crosses 180 at row 20, alert fires at row 16 (60 min earlier on 15-min rows)
    n = 40
    g = np.r_[np.full(20, 150.0), np.full(20, 200.0)]
    te = pd.DataFrame({"patient_id": "P", "step_idx": np.arange(n) * 3, "g_now": g})
    prob = np.zeros(n)
    prob[16:20] = 0.9
    m = ev.alert_metrics(te, prob, 0.5)
    assert m["excursions"] == 1 and m["detected_pct"] == 100 and m["median_lead_min"] == 60
    assert m["false_alert_share_pct"] == 0


def test_perturbations_degrade_inputs():
    ehr, s = _small(3, 3)
    assert ev.perturb(s, "no_meal_log").meal_carbs_g.isna().all()
    assert ev.perturb(s, "cgm_dropout_20pct").glucose.isna().mean() > 0.1
    with pytest.raises(ValueError):
        ev.perturb(s, "bogus")


# ------------------------------------------------------------------ synthea importer (mock CSVs in Synthea layout)
def test_synthea_importer_on_mock_csvs(tmp_path):
    from src.import_synthea import import_synthea
    pd.DataFrame({"Id": ["a", "b", "c"], "BIRTHDATE": ["1970-01-01", "1980-05-05", "1990-02-02"], "GENDER": ["F", "M", "F"]}).to_csv(tmp_path / "patients.csv", index=False)
    pd.DataFrame({"PATIENT": ["a", "b", "c", "a"], "START": ["2015-01-01", "2018-01-01", "2019-01-01", "2016-01-01"],
                  "DESCRIPTION": ["Diabetes mellitus type 2", "Prediabetes", "Diabetes mellitus type 2", "Hypertension"]}).to_csv(tmp_path / "conditions.csv", index=False)
    pd.DataFrame({"PATIENT": ["a", "a", "c"], "DATE": ["2024-01-01", "2024-06-01", "2024-06-01"], "CODE": ["4548-4", "4548-4", "4548-4"],
                  "VALUE": [7.0, 8.2, 6.8]}).to_csv(tmp_path / "observations.csv", index=False)
    out = import_synthea(str(tmp_path))
    assert len(out) == 2 and set(out.columns) >= set(generate_ehr(3).columns)
    assert out.loc[0, "hba1c"] == 8.2 and out.loc[0, "hypertension"] == 1           # latest HbA1c, hypertension flag
    assert build_features(simulate_cohort(out, days=3, seed=0), out).shape[0] > 0     # imported EHR flows through the whole pipeline


# ------------------------------------------------------------------ twin
@pytest.fixture(scope="module")
def twin_parts():
    ehr = pd.read_csv("data/demo_ehr.csv")
    s = pd.read_csv("data/demo_sensors.csv.gz", parse_dates=["ts"])
    return ehr, s, load_bundle()


def test_twin_forecast_explain_and_what_if(twin_parts):
    ehr, s, b = twin_parts
    e = ehr.iloc[-1]
    tw = DigitalTwin.from_history(e, s[s.patient_id == e.patient_id].iloc[:900], b)
    fc = tw.forecast()
    assert fc["lo"] <= fc["pred_120"] <= fc["hi"] and 0 <= fc["spike_prob"] <= 1 and 0 <= fc["hypo_prob"] <= 1
    ex = tw.explain()
    assert set(ex.index) == set(b["groups"]) and ex.notna().all().all()
    wi = tw.what_if(planned_carbs=80)
    eat, skip = wi["curves"]["Eat 80 g carbs"], wi["curves"]["Skip the meal"]
    assert eat.max() > skip.max() and wi["curves"]["Eat + 20 min walk after"].max() < eat.max()


def test_twin_degrades_gracefully_and_reports_it(twin_parts):
    ehr, s, b = twin_parts
    e = ehr.iloc[0]
    h = s[s.patient_id == e.patient_id].iloc[:900].copy()
    h[["steps", "hr", "hrv"]] = np.nan
    h["sleep_stage"] = np.nan
    h.iloc[-9:, h.columns.get_loc("glucose")] = np.nan
    tw = DigitalTwin.from_history(e, h, b)
    fc = tw.forecast()
    assert np.isfinite(fc["pred_120"])
    issues = " ".join(tw.data_quality()["issues"])
    assert "Wearables offline" in issues and "CGM gap" in issues
