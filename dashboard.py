"""DiaTwin clinician dashboard.   Run:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import HYPO_MGDL, SPIKE_MGDL  # noqa: E402
from src.twin import DigitalTwin, load_bundle  # noqa: E402

st.set_page_config(page_title="DiaTwin - Type 2 Diabetes Digital Twin", page_icon="🩺", layout="wide")

NAVY, TEAL, CORAL, AMBER, MID = "#14213D", "#0E9F8E", "#E4572E", "#F2A541", "#5C8EA8"
LEVEL_COLOR = {"LOW": TEAL, "OK": TEAL, "WATCH": AMBER, "ALERT": CORAL, "HIGH NOW": CORAL, "LOW NOW": CORAL}


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_data():
    ehr = pd.read_csv(ROOT / "data" / "demo_ehr.csv")
    sens = pd.read_csv(ROOT / "data" / "demo_sensors.csv.gz", parse_dates=["ts"])
    return ehr, sens


@st.cache_data
def get_metrics():
    return json.loads((ROOT / "results" / "metrics.json").read_text())


bundle = get_bundle()
ehr_all, sens_all = get_data()
metrics = get_metrics()
FIG = ROOT / "results" / "figures"


def degrade(series: pd.DataFrame, idx: int, wearables_off: bool, meals_off: bool, cgm_gap: bool) -> pd.DataFrame:
    """Data-quality simulator: what happens to the twin when an input stream breaks."""
    s = series.iloc[: idx + 1].copy()
    if wearables_off:
        s[["steps", "hr", "hrv"]] = np.nan
        s["sleep_stage"] = np.nan
    if meals_off:
        s["meal_carbs_g"] = np.nan
    if cgm_gap:
        s.iloc[-9:, s.columns.get_loc("glucose")] = np.nan          # last 45 minutes
    return s


def tile(label, level, msg):
    color = LEVEL_COLOR.get(level, MID)
    return (f"<div style='background:{color};color:white;padding:12px 14px;border-radius:10px;font-weight:700;"
            f"text-align:center'>{label}: {level}<br><span style='font-weight:400;font-size:0.78rem'>{msg}</span></div>")


st.title("🩺 DiaTwin - Type 2 Diabetes Digital Twin")
st.caption("Proof-of-concept on **synthetic** data (EHR + wearable/CGM streams). Not a medical device; not medical advice.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Virtual patient")
    labels = {r.patient_id: f"{r.patient_id} · {r.age}{r.sex} · HbA1c {r.hba1c}% · {r.medication}" for r in ehr_all.itertuples()}
    pid = st.selectbox("Patient", list(labels), format_func=labels.get)
    series = sens_all[sens_all.patient_id == pid].reset_index(drop=True)
    first, last = 288 + 12, len(series) - 26
    idx = st.slider("Replay time (live stream position)", first, last, min(first + 288 * 3 + 100, last),
                    help="Moves the twin along the simulated sensor stream, as if the data were arriving live.")
    st.caption(f"Now: **{series.ts[idx]:%a %d %b, %H:%M}**")
    reveal = st.checkbox("Reveal what actually happened (next 2 h)", value=False)
    st.divider()
    st.header("What-if planner")
    carbs = st.slider("Planned meal carbs (g)", 20, 120, 70, 5)
    in_min = st.slider("Meal starts in (min)", 0, 60, 15, 5)
    walk = st.slider("Walk after meal (min)", 10, 40, 20, 5)
    st.divider()
    st.header("Data-quality simulator")
    off_w = st.checkbox("Wearables offline")
    off_m = st.checkbox("Meal diary not used")
    gap = st.checkbox("CGM dropout (last 45 min)")

ehr = ehr_all[ehr_all.patient_id == pid].iloc[0]
twin = DigitalTwin.from_history(ehr, degrade(series, idx, off_w, off_m, gap), bundle)
fc = twin.forecast()
dq = twin.data_quality()

tabs = st.tabs(["Twin view", "Cohort board", "Data fusion", "Model performance", "Safety & limits"])

# ======================= TAB 1: twin view =======================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Glucose now", f"{fc['now']:.0f} mg/dL")
    c2.metric("+30 / +60 min", f"{fc['pred_30']:.0f} / {fc['pred_60']:.0f}")
    c3.metric("+120 min", f"{fc['pred_120']:.0f} mg/dL", f"{fc['pred_120'] - fc['now']:+.0f}")
    c4.metric("Spike-onset risk", f"{fc['spike_prob'] * 100:.0f}%")
    t1, t2 = st.columns(2)
    t1.markdown(tile("Hyperglycaemia", fc["level"], fc["message"]), unsafe_allow_html=True)
    t2.markdown(tile("Hypoglycaemia", fc["hypo_level"], fc["hypo_message"]), unsafe_allow_html=True)
    for issue in dq["issues"]:
        st.warning(issue)

    hist = series.iloc[max(0, idx - 72): idx + 1]
    t0 = fc["ts"]
    fig = go.Figure()
    fig.add_hrect(y0=HYPO_MGDL, y1=SPIKE_MGDL, fillcolor=TEAL, opacity=0.07, line_width=0)
    fig.add_trace(go.Scatter(x=hist.ts, y=hist.glucose, name="CGM (last 6 h)", line=dict(color=NAVY, width=2.5)))
    xs = [t0 + pd.Timedelta(minutes=m) for m in (0, 30, 60, 120)]
    fig.add_trace(go.Scatter(x=[xs[0], xs[3], xs[3], xs[0]], y=[fc["now"], fc["hi"], fc["lo"], fc["now"]], fill="toself",
                             fillcolor="rgba(14,159,142,0.2)", line=dict(width=0), name="80% band (2 h)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[fc["now"], fc["pred_30"], fc["pred_60"], fc["pred_120"]], name="Twin forecast",
                             line=dict(color=TEAL, width=3, dash="dot"), mode="lines+markers"))
    if reveal:
        fut = series.iloc[idx: idx + 25]
        fig.add_trace(go.Scatter(x=fut.ts, y=fut.glucose_true, name="What actually happened", line=dict(color=CORAL, width=2)))
    fig.add_hline(y=SPIKE_MGDL, line_dash="dash", line_color=CORAL, annotation_text="180")
    fig.add_hline(y=HYPO_MGDL, line_dash="dash", line_color=CORAL, annotation_text="70")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="mg/dL",
                      legend=dict(orientation="h", y=1.14), title="Glucose and multi-horizon forecast")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        wi = twin.what_if(carbs, in_min, walk)
        fig2 = go.Figure()
        for (name, y), col in zip(wi["curves"].items(), [CORAL, AMBER, TEAL, NAVY]):
            fig2.add_trace(go.Scatter(x=wi["minutes"], y=y, name=name, line=dict(color=col, width=2.5)))
        fig2.add_hline(y=SPIKE_MGDL, line_dash="dash", line_color=CORAL)
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Minutes from now", yaxis_title="mg/dL",
                           title="What-if: next 3 hours under different choices", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig2, width="stretch")
        st.caption("What-if curves come from the twin's personalised mechanistic layer (insulin sensitivity derived from the "
                   "patient's EHR). Illustrative on synthetic data.")
    with right:
        st.subheader("Why this forecast?")
        ex = twin.explain()
        fig3 = go.Figure(go.Bar(x=ex["effect_on_2h_change_mgdl"], y=[s.split(" (")[0] for s in ex.index], orientation="h",
                                marker_color=[CORAL if v > 0 else TEAL for v in ex["effect_on_2h_change_mgdl"]]))
        fig3.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Effect on predicted 2 h change (mg/dL)")
        st.plotly_chart(fig3, width="stretch")
        st.caption("Each bar: how much that data group moves the forecast versus a typical patient. Red pushes glucose up, green down.")
        st.subheader("Clinical summary")
        for note in twin.summary():
            st.markdown(f"- {note}")

# ======================= TAB 2: cohort board =======================
with tabs[1]:
    st.subheader("Who needs attention right now?")
    st.caption("All 12 demo patients at the same replay position, ranked by risk. A clinician's morning-round view.")

    @st.cache_data(show_spinner=False)
    def cohort_board(i):
        rows = []
        for e in ehr_all.itertuples(index=False):
            ser = sens_all[sens_all.patient_id == e.patient_id].reset_index(drop=True)
            f = DigitalTwin.from_history(pd.Series(e._asdict()), ser.iloc[: i + 1], bundle).forecast()
            rows.append({"Patient": e.patient_id, "HbA1c %": e.hba1c, "Medication": e.medication, "Now": round(f["now"]),
                         "+2 h": round(f["pred_120"]), "Spike risk %": round(f["spike_prob"] * 100),
                         "Hyper status": f["level"], "Hypo risk": f["hypo_level"], "_rank": f["spike_prob"] + 0.5 * f["hypo_prob"]})
        return pd.DataFrame(rows).sort_values("_rank", ascending=False).drop(columns="_rank")
    board = cohort_board(idx)
    st.dataframe(board.style.map(lambda v: f"color:{LEVEL_COLOR.get(v, 'inherit')};font-weight:700" if v in LEVEL_COLOR else "",
                                 subset=["Hyper status", "Hypo risk"]), width="stretch", hide_index=True)

# ======================= TAB 3: data fusion =======================
with tabs[2]:
    a, b = st.columns([1, 2])
    with a:
        st.subheader("Stream 1 - static EHR")
        prof = {"Age / sex": f"{ehr.age} / {ehr.sex}", "BMI": ehr.bmi, "Diabetes duration": f"{ehr.duration_years} y",
                "HbA1c": f"{ehr.hba1c} %", "Fasting glucose": f"{ehr.fasting_glucose} mg/dL", "eGFR": ehr.egfr, "LDL": ehr.ldl,
                "Hypertension": "yes" if ehr.hypertension else "no", "TCF7L2 risk alleles": ehr.tcf7l2_risk_alleles,
                "Medication": ehr.medication, "Activity": ehr.activity_level}
        st.dataframe(pd.Series(prof, name="value").astype(str).to_frame(), width="stretch")
        st.metric("Personal insulin sensitivity", f"{wi['insulin_sensitivity']:.2f}", help="1.0 = average patient")
    with b:
        st.subheader("Stream 2 - dynamic wearables (last 24 h)")
        day = twin.buf.tail(288)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=day.ts, y=day.hr, name="Heart rate (bpm)", line=dict(color=CORAL)))
        fig4.add_trace(go.Scatter(x=day.ts, y=day.hrv, name="HRV RMSSD (ms)", line=dict(color=TEAL)))
        fig4.add_trace(go.Bar(x=day.ts, y=day.steps, name="Steps / 5 min", marker_color="#9DB4C0", yaxis="y2", opacity=0.6))
        meals = day[day.meal_carbs_g.notna()]
        fig4.add_trace(go.Scatter(x=meals.ts, y=[20] * len(meals), mode="markers+text", text=meals.meal_carbs_g.astype(int).astype(str) + " g",
                                  textposition="top center", name="Logged meal", marker=dict(size=11, color=AMBER, symbol="diamond")))
        fig4.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12),
                           yaxis2=dict(overlaying="y", side="right", showgrid=False))
        st.plotly_chart(fig4, width="stretch")
    st.subheader("Fused feature vector fed to the model (latest)")
    row = twin.fused_row()[bundle["features"]].T
    row.columns = ["value"]
    st.dataframe(row.round(2), width="stretch", height=260)
    q1, q2, q3 = st.columns(3)
    q1.metric("CGM gap", f"{dq['cgm_gap_min']} min")
    q2.metric("Wearables online", "yes" if dq["wearables_online"] else "NO")
    q3.metric("Meals logged (24 h)", dq["meals_logged_24h"])

# ======================= TAB 4: performance =======================
with tabs[3]:
    R, C, D = metrics["regression"], metrics["classification"], metrics["dataset"]
    B, A, HY = metrics["bootstrap"], metrics["alerts"]["fusion"], metrics["hypoglycaemia"]["fusion"]
    st.markdown(f"Evaluated on **{D['train_val_test_patients'][2]} held-out virtual patients** (patient-level split; {D['patients']} patients x "
                f"{D['days_per_patient']} days of 5-min data, {D['sensor_rows']:,} readings). All intervals are patient-level bootstrap 95% CIs.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast MAE (2 h)", f"{R['fusion']['mae']:.1f} mg/dL", f"{R['fusion']['mae'] - R['persistence']['mae']:+.1f} vs persistence", delta_color="inverse")
    m2.metric("Spike-onset ROC-AUC", f"{C['fusion']['roc_auc']:.3f}", f"CI {B['spike_auc']['fusion']['ci95'][0]:.3f}-{B['spike_auc']['fusion']['ci95'][1]:.3f}", delta_color="off")
    m3.metric("Excursions flagged early", f"{A['detected_pct']:.0f}%", f"median lead {A['median_lead_min']:.0f} min", delta_color="off")
    m4.metric("Hypo-alert ROC-AUC", f"{HY['roc_auc']:.3f}", "rare event", delta_color="off")

    sub = st.tabs(["Accuracy", "Alerts", "Subgroups", "Robustness & shift", "Explainability", "Cross-validation"])
    with sub[0]:
        c1, c2 = st.columns(2)
        c1.image(str(FIG / "ablation_mae.png"))
        c2.image(str(FIG / "cold_start_mae.png"))
        c3, c4 = st.columns(2)
        c3.image(str(FIG / "horizon_mae.png"))
        c4.image(str(FIG / "model_families.png"))
        st.image(str(FIG / "forecast_example.png"))
    with sub[1]:
        c1, c2 = st.columns(2)
        c1.image(str(FIG / "roc_spike_alert.png"))
        c2.image(str(FIG / "calibration_spike.png"))
        c3, c4 = st.columns(2)
        c3.image(str(FIG / "alert_lead_time.png"))
        c4.image(str(FIG / "operating_points.png"))
        st.markdown("**Operating points** - a higher threshold means fewer alerts but more missed excursions; the clinician picks the trade-off.")
        st.dataframe(pd.DataFrame(metrics["operating_points"]).round(2), width="stretch", hide_index=True)
        hy = metrics["hypoglycaemia"]
        st.markdown(f"**Hypoglycaemia alert** (rare event: {metrics['dataset']['hypo_positive_rows_test']} positive rows from "
                    f"{metrics['dataset']['hypo_positive_patients_test']} test patients): fusion ROC-AUC {hy['fusion']['roc_auc']:.3f} vs CGM-only {hy['cgm_only']['roc_auc']:.3f}; "
                    f"recall {hy['fusion']['recall'] * 100:.0f}% at precision {hy['fusion']['precision'] * 100:.0f}%.")
    with sub[2]:
        st.image(str(FIG / "subgroup_mae.png"))
        st.dataframe(pd.DataFrame(metrics["subgroups"]).round(2), width="stretch", hide_index=True)
    with sub[3]:
        st.image(str(FIG / "robustness_mae.png"))
        sh = metrics["distribution_shift"]
        st.markdown(f"**Distribution shift** - unseen cohort of {sh['cohort']['patients']} patients, older ({sh['cohort']['mean_age']:.0f} vs "
                    f"{sh['cohort']['train_mean_age']:.0f} y), heavier (BMI {sh['cohort']['mean_bmi']:.1f} vs {sh['cohort']['train_mean_bmi']:.1f}), "
                    f"longer disease duration ({sh['cohort']['mean_duration_y']:.1f} vs {sh['cohort']['train_mean_duration_y']:.1f} y): "
                    f"persistence MAE {sh['persistence_mae']:.1f}, CGM-only {sh['cgm_only_mae']:.1f}, **fusion {sh['fusion_mae']:.1f} mg/dL**.")
    with sub[4]:
        c1, c2 = st.columns(2)
        c1.image(str(FIG / "feature_importance.png"))
        c2.image(str(FIG / "group_importance.png"))
    with sub[5]:
        cv = metrics["cross_validation"]
        st.dataframe(pd.DataFrame({k: {"mean MAE": round(v["mean"], 2), "sd": round(v["sd"], 2)} for k, v in cv.items()}).T, width="stretch")
        st.caption("5-fold GroupKFold over all patients (every fold holds out whole patients).")
    st.info("Honest reading: with a full day of CGM history, EHR and wearables add a modest gain because glucose history already reveals "
            "the patient's baseline. The big wins are cold start, robustness to missing streams (modality-dropout training) and personalised what-if.")

# ======================= TAB 5: safety & limits =======================
with tabs[4]:
    st.subheader("Safety, ethics and limits")
    st.markdown("""
- **Synthetic data only.** No real patient data. Accuracy figures describe our simulator, not clinical performance.
- **Not a medical device.** Alerts are decision support for a clinician, never an autonomous treatment decision.
- **Hypoglycaemia is rare** in the cohort; the hypo alert is a ranking score tuned for recall and must be validated on real data before any use.
- **What-if layer shares the simulator's physiology**, so its real-world fidelity is untested.
- **Fairness:** performance is reported per HbA1c band, medication, age, sex and activity level (Model performance, Subgroups).
- **Graceful degradation:** the twin flags missing CGM, offline wearables and unlogged meals instead of silently guessing.
- **Privacy:** all data are generated locally; nothing leaves the machine.
""")
