"""Generates docs/architecture_diagram.pdf (2 pages) plus PNGs.   python docs/make_architecture.py

Page 1: system architecture (streams -> fusion -> twin core -> clinician outputs)
Page 2: ML pipeline and evaluation flow (data -> features -> training -> 12-part evaluation -> artefacts -> app)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
NAVY, TEAL, CORAL, AMBER, SOFT, MID = "#14213D", "#0E9F8E", "#E4572E", "#F2A541", "#EEF3F7", "#5C8EA8"


def canvas():
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), fc="white", ec=NAVY, tc=NAVY, ts=11.5, bs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=tc)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.3 - i * 2.1, ln, ha="center", va="top", fontsize=bs, color=tc)


def arrow(ax, x1, y1, x2, y2, color=NAVY, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def header(ax, title, sub):
    ax.text(2, 54, title, fontsize=20, fontweight="bold", color=NAVY, va="center")
    ax.text(2, 51.2, sub, fontsize=11, color="#4A5568", va="center")


def page1():
    fig, ax = canvas()
    header(ax, "DiaTwin - System architecture", "Type 2 Diabetes digital twin: fusion of static EHR and dynamic wearable/CGM streams")
    for x, t in [(2, "1  DATA STREAMS"), (28, "2  FUSION"), (50, "3  TWIN CORE"), (79, "4  CLINICIAN OUTPUT")]:
        ax.text(x, 47.5, t, fontsize=10, fontweight="bold", color=TEAL)
    box(ax, 2, 31, 22, 14, "Static / historical", ["Synthea-style simulated EHR", "demographics, HbA1c, eGFR, LDL", "TCF7L2 marker, medication", "comorbidities, lifestyle"], fc=SOFT)
    box(ax, 2, 12, 22, 16, "Dynamic / real-time", ["CGM glucose (5 min, with gaps)", "heart rate, HRV (RMSSD)", "steps, sleep stages", "meal diary (logged carbs)"], fc=SOFT)
    box(ax, 28, 17, 18, 24, "Fusion layer", ["align streams per patient", "glucose lags, trends, stats", "sleep / HRV / activity", "meal timing, time of day", "+ static EHR features", "= 43-feature vector", "missing streams stay NaN"], ec=TEAL)
    arrow(ax, 24.3, 38, 27.7, 33)
    arrow(ax, 24.3, 20, 27.7, 25)
    box(ax, 50, 31, 25, 14, "Learned layer", ["Gradient-boosted trees", "30 / 60 / 120 min forecasts", "10-90% band, spike + hypo risk", "modality-dropout training"], ec=TEAL)
    box(ax, 50, 12, 25, 15, "Mechanistic layer", ["Personal insulin sensitivity", "(derived from the EHR)", "meal + exercise kernels", "what-if simulation"], ec=AMBER)
    arrow(ax, 46.3, 33, 49.7, 37)
    arrow(ax, 46.3, 25, 49.7, 20)
    ax.text(62.5, 28.6, "DIGITAL TWIN  (per patient)", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.add_patch(FancyBboxPatch((48.6, 10.4), 27.8, 36, boxstyle="round,pad=0.2,rounding_size=1.2", fc="none", ec=NAVY, lw=1.2, ls="--"))
    box(ax, 79, 38, 19, 7, "Alerts", ["hyper / hypo, LOW-WATCH-ALERT"], fc=NAVY, ec=NAVY, tc="white", ts=10.5, bs=8.5)
    box(ax, 79, 29.5, 19, 7, "Explanations", ["why: feature-group effects"], fc=NAVY, ec=NAVY, tc="white", ts=10.5, bs=8.5)
    box(ax, 79, 21, 19, 7, "What-if planner", ["eat / half / walk / skip"], fc=NAVY, ec=NAVY, tc="white", ts=10.5, bs=8.5)
    box(ax, 79, 12.5, 19, 7, "Cohort + data quality", ["risk board, stream health"], fc=NAVY, ec=NAVY, tc="white", ts=10.5, bs=8.5)
    for y in (41.5, 33, 24.5, 16):
        arrow(ax, 76.6, 28, 78.7, y)
    ax.add_patch(FancyBboxPatch((2, 1.2), 96, 6.6, boxstyle="round,pad=0.2,rounding_size=1.0", fc=SOFT, ec="none"))
    for x, t, s in [(4, "Streamlit dashboard", "doctor-facing virtual-patient app"), (34, "Rigorous evaluation", "patient-level split, CIs, subgroups, robustness"), (68, "Privacy by design", "synthetic data only (DPDP Act / HIPAA safe)")]:
        ax.text(x, 5.3, t, fontsize=9.5, fontweight="bold", color=NAVY, va="center")
        ax.text(x, 3.0, s, fontsize=8.5, color="#4A5568", va="center")
    return fig


def page2():
    fig, ax = canvas()
    header(ax, "DiaTwin - ML pipeline and evaluation", "Reproducible with one command: python -m src.train")
    ax.text(2, 47.5, "BUILD", fontsize=10, fontweight="bold", color=TEAL)
    box(ax, 2, 33, 17, 12, "1  Synthetic data", ["400 patients x 14 days", "EHR + physiology simulator", "drug effect, dropouts"], fc=SOFT, ts=10.5, bs=8.5)
    box(ax, 22, 33, 17, 12, "2  Features", ["43 fused features", "targets: 30/60/120 min,", "spike, hypo (true glucose)"], fc=SOFT, ts=10.5, bs=8.5)
    box(ax, 42, 33, 17, 12, "3  Patient split", ["60 / 20 / 20 patients", "no patient in two sets", "first 24 h = history"], fc=SOFT, ts=10.5, bs=8.5)
    box(ax, 62, 33, 17, 12, "4  Training", ["HistGradientBoosting", "modality-dropout aug.", "thresholds on validation"], fc=SOFT, ts=10.5, bs=8.5)
    box(ax, 82, 33, 16, 12, "5  Artefacts", ["models/*.joblib", "metrics.json, figures", "demo patients"], fc=SOFT, ts=10.5, bs=8.5)
    for x in (19.3, 39.3, 59.3, 79.3):
        arrow(ax, x, 39, x + 2.4, 39)
    ax.text(2, 29.5, "EVALUATION SUITE (all patient-level)", fontsize=10, fontweight="bold", color=TEAL)
    items = [("Ablation", "CGM / +wearables / +EHR / fusion"), ("Cold start", "new patient, < 1 h data"), ("Model families", "ridge, extra-trees, MLP, GBM"),
             ("Multi-horizon", "30 / 60 / 120 min"), ("Bootstrap CIs", "patient-level, paired gains"), ("Calibration", "Brier, ECE, reliability"),
             ("Alert burden", "episodes/day, lead time"), ("Subgroups", "HbA1c, drug, age, sex, activity"), ("Robustness", "noise, dropout, no diary, no wearables"),
             ("Distribution shift", "older / heavier / longer duration"), ("5-fold CV", "GroupKFold over patients"), ("Importance", "feature + group permutation")]
    for i, (t, s) in enumerate(items):
        x = 2 + (i % 4) * 24.2
        y = 21.2 - (i // 4) * 7.4
        box(ax, x, y, 22.5, 6.3, f"{i + 1}  {t}", [s], fc="white", ec=MID, ts=10, bs=8.5)
    ax.add_patch(FancyBboxPatch((2, 0.6), 96, 3.8, boxstyle="round,pad=0.2,rounding_size=1.0", fc=SOFT, ec="none"))
    ax.text(4, 2.5, "Serving: Streamlit dashboard loads the saved models and replays each virtual patient's stream.   CI: GitHub Actions runs the test-suite on every push.",
            fontsize=9, color=NAVY, va="center")
    return fig


if __name__ == "__main__":
    f1, f2 = page1(), page2()
    with PdfPages(OUT / "architecture_diagram.pdf") as pdf:
        pdf.savefig(f1, bbox_inches="tight")
        pdf.savefig(f2, bbox_inches="tight")
    f1.savefig(OUT / "architecture_diagram.png", dpi=170, bbox_inches="tight")
    f2.savefig(OUT / "architecture_pipeline.png", dpi=170, bbox_inches="tight")
    print("saved")
