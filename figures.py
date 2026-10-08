"""Result figures used in the README, technical report and presentation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import HYPO_MGDL, SPIKE_MGDL

NAVY, TEAL, CORAL, AMBER, GREY, MID = "#14213D", "#0E9F8E", "#E4572E", "#F2A541", "#8A94A6", "#5C8EA8"
LABELS = {"cgm_only": "CGM only", "cgm+wearables": "CGM + wearables", "cgm+ehr": "CGM + EHR", "fusion": "Full fusion"}


def _style():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": NAVY, "ytick.color": NAVY, "figure.dpi": 150})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def _bar_labels(ax, bars, vals, fmt="{:.1f}", dy=0.3):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v), ha="center", fontweight="bold", fontsize=10)


def fig_ablation(out, M):
    R, B = M["regression"], M["bootstrap"]["mae"]
    keys = ["persistence", "cgm_only", "cgm+wearables", "cgm+ehr", "fusion"]
    names = ["Persistence"] + [LABELS[k].replace(" + ", "\n+ ") for k in keys[1:]]
    vals = [R[k]["mae"] for k in keys]
    ci = {k: B[k]["ci95"] for k in ("persistence", "cgm_only", "fusion")}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(names, vals, color=[GREY, "#9DB4C0", MID, MID, TEAL], width=0.62)
    for k, b in zip(keys, bars):
        if k in ci:
            ax.errorbar(b.get_x() + b.get_width() / 2, R[k]["mae"], yerr=[[R[k]["mae"] - ci[k][0]], [ci[k][1] - R[k]["mae"]]],
                        color=NAVY, capsize=4, lw=1.4)
    _bar_labels(ax, bars, vals, dy=0.9)
    ax.set_ylabel("MAE of 2-hour forecast (mg/dL)")
    ax.set_title("Forecast error, with 95% patient-level bootstrap CI", loc="left", fontweight="bold")
    _save(fig, out, "ablation_mae.png")


def fig_cold(out, M):
    c = M["cold_start"]
    keys = ["cold_cgm", "cold_cgm+wearables", "cold_cgm+ehr", "cold_fusion"]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [c[k]["mae"] for k in keys]
    bars = ax.bar(["CGM\n(<1 h)", "CGM\n+ wearables", "CGM\n+ EHR", "Full\nfusion"], vals, color=["#9DB4C0", MID, MID, TEAL], width=0.6)
    _bar_labels(ax, bars, vals)
    ax.set_ylabel("MAE of 2-hour forecast (mg/dL)")
    ax.set_title("Cold start: a new patient with < 1 h of data", loc="left", fontweight="bold")
    _save(fig, out, "cold_start_mae.png")


def fig_horizon(out, M):
    H = M["horizons"]
    hs = ["30", "60", "120"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, c, lbl, ls in (("persistence", GREY, "Persistence", "--"), ("cgm_only", MID, "CGM only", "-"), ("fusion", TEAL, "Full fusion", "-")):
        ax.plot([int(h) for h in hs], [H[h][k]["mae"] for h in hs], marker="o", color=c, lw=2.4, ls=ls, label=lbl)
    ax.set_xticks([30, 60, 120])
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Error grows with horizon; the twin flattens it", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, out, "horizon_mae.png")


def fig_families(out, M):
    F = M["model_families"]
    names = list(F)
    vals = [F[k]["mae"] for k in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(names[::-1], vals[::-1], color=[TEAL if "ours" in n else GREY for n in names[::-1]])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontweight="bold")
    ax.set_xlabel("MAE of 2-hour forecast (mg/dL), same fusion features")
    ax.set_title("Model-family comparison", loc="left", fontweight="bold")
    _save(fig, out, "model_families.png")


def fig_importance(out, imp, gimp):
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(top.index, top.values, color=TEAL)
    ax.set_xlabel("Permutation importance (increase in MAE, mg/dL)")
    ax.set_title("What drives the forecast: top features", loc="left", fontweight="bold")
    _save(fig, out, "feature_importance.png")
    g = gimp[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh([s.replace(" (", "\n(") for s in g.index], g.values, color=[MID, MID, AMBER, TEAL][-len(g):][::-1] if False else TEAL)
    ax.set_xlabel("Permutation importance of the whole group (increase in MAE, mg/dL)")
    ax.set_title("What drives the forecast: feature groups", loc="left", fontweight="bold")
    _save(fig, out, "group_importance.png")


def fig_roc(out, te_o, prob_o):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for k, c in [("cgm_only", GREY), ("fusion", TEAL)]:
        fpr, tpr, _ = roc_curve(te_o.y_spike.values, prob_o[k])
        ax.plot(fpr, tpr, color=c, lw=2.2, label=LABELS[k])
    ax.plot([0, 1], [0, 1], ls="--", color="#CCD3DD")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Spike-onset alert (ROC)", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, out, "roc_spike_alert.png")


def fig_calibration(out, M):
    c = M["calibration"]["spike"]
    pts = np.array(c["curve"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#CCD3DD", label="Perfect calibration")
    ax.plot(pts[:, 0], pts[:, 1], marker="o", color=TEAL, lw=2.2, label="DiaTwin")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (Brier {c['brier']:.3f}, ECE {c['ece']:.3f})", loc="left", fontweight="bold", fontsize=11)
    ax.legend(frameon=False)
    _save(fig, out, "calibration_spike.png")


def fig_lead(out, M):
    leads = M["alerts"]["fusion"]["lead_times_min"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(leads, bins=np.arange(0, 135, 15), color=TEAL, edgecolor="white")
    ax.set_xlabel("Warning time before glucose crosses 180 mg/dL (minutes)")
    ax.set_ylabel("Excursions")
    a = M["alerts"]["fusion"]
    ax.set_title(f"Lead time (median {a['median_lead_min']:.0f} min; {a['detected_pct']:.1f}% of excursions flagged)", loc="left", fontweight="bold", fontsize=11)
    ax.text(0.02, 0.9, "last bar = 120 min\n(capped by the 2 h look-back)", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=GREY)
    _save(fig, out, "alert_lead_time.png")


def fig_operating_points(out, M):
    ops = M["operating_points"]
    t = [o["threshold"] for o in ops]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(t, [o["detected_with_30min_lead_pct"] for o in ops], marker="o", color=TEAL, lw=2.4, label="Excursions flagged >= 30 min ahead (%)")
    ax.plot(t, [o["false_alert_share_pct"] for o in ops], marker="s", color=CORAL, lw=2.4, label="False-alert share (%)")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(t, [o["alert_episodes_per_patient_day"] for o in ops], marker="^", color=NAVY, lw=2, ls="--", label="Alert episodes / patient-day")
    ax.set_xlabel("Alert probability threshold")
    ax.set_ylabel("%")
    ax2.set_ylabel("Episodes per patient-day")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    ax.set_title("Choosing an operating point: sensitivity vs alert fatigue", loc="left", fontweight="bold", fontsize=11)
    _save(fig, out, "operating_points.png")


def fig_subgroups(out, M):
    rows = [r for r in M["subgroups"] if r["patients"] >= 5]
    lab = [f"{r['dimension']}: {r['group']}" for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.4))
    ax.barh(y + 0.2, [r["mae_persistence"] for r in rows], 0.38, color=GREY, label="Persistence")
    ax.barh(y - 0.2, [r["mae_fusion"] for r in rows], 0.38, color=TEAL, label="Full fusion")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (mg/dL)")
    ax.set_title("Subgroup analysis: does the twin work for everyone?", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.08))
    _save(fig, out, "subgroup_mae.png")


def fig_robustness(out, M):
    Rb = M["robustness"]
    kinds = list(Rb)
    nice = {"clean": "Clean", "cgm_noise_8mgdl": "CGM noise\n+8 mg/dL", "cgm_dropout_20pct": "20% CGM\ndropout", "no_meal_log": "No meal\ndiary", "wearables_offline": "Wearables\noffline"}
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.2
    ax.bar(x - 1.5 * w, [Rb[k]["persistence_mae"] for k in kinds], w, color=GREY, label="Persistence")
    ax.bar(x - 0.5 * w, [Rb[k]["cgm_only_mae"] for k in kinds], w, color=MID, label="CGM only")
    ax.bar(x + 0.5 * w, [Rb[k]["fusion_plain_mae"] for k in kinds], w, color=AMBER, label="Fusion (plain training)")
    b = ax.bar(x + 1.5 * w, [Rb[k]["fusion_mae"] for k in kinds], w, color=TEAL, label="Fusion (modality-dropout)")
    _bar_labels(ax, b, [Rb[k]["fusion_mae"] for k in kinds], dy=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[k] for k in kinds])
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Robustness to degraded inputs (models not retrained)", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _save(fig, out, "robustness_mae.png")


def fig_agp(out, agp):
    t = agp.index / 60.0
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.fill_between(t, agp.p05, agp.p95, color=TEAL, alpha=0.15, label="5th-95th percentile")
    ax.fill_between(t, agp.p25, agp.p75, color=TEAL, alpha=0.35, label="25th-75th percentile")
    ax.plot(t, agp.p50, color=NAVY, lw=2.4, label="Median")
    ax.axhspan(HYPO_MGDL, SPIKE_MGDL, color=GREY, alpha=0.08)
    ax.axhline(SPIKE_MGDL, color=CORAL, ls="--", lw=1)
    ax.axhline(HYPO_MGDL, color=CORAL, ls="--", lw=1)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Ambulatory Glucose Profile of the simulated cohort", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _save(fig, out, "agp_cohort.png")


def fig_forecast(out, te, abs_pred, lo, hi):
    d = te.assign(pred=abs_pred, lo=lo, hi=hi)
    pid = d.groupby("patient_id").y_g120.max().sort_values().index[-4]
    d = d[d.patient_id == pid].sort_values("ts")
    d = d[d.ts >= d.ts.min() + pd.Timedelta(hours=6)].head(96)
    tt = d.ts + pd.Timedelta(minutes=120)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(tt, d.lo, d.hi, color=TEAL, alpha=0.18, label="80% prediction band")
    ax.plot(tt, d.y_g120, color=NAVY, lw=1.8, label="Actual CGM")
    ax.plot(tt, d.pred, color=TEAL, lw=2, label="Twin forecast (made 2 h earlier)")
    ax.axhline(SPIKE_MGDL, color=CORAL, ls="--", lw=1.2)
    ax.text(tt.iloc[0], SPIKE_MGDL + 4, "180 mg/dL", color=CORAL, fontsize=9)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"Held-out patient {pid}: forecast vs reality", loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _save(fig, out, "forecast_example.png")


def make_all(out, ctx):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    M = ctx["M"]
    fig_ablation(out, M)
    fig_cold(out, M)
    fig_horizon(out, M)
    fig_families(out, M)
    fig_importance(out, ctx["imp"], ctx["gimp"])
    fig_roc(out, ctx["te_o"], ctx["prob_o"])
    fig_calibration(out, M)
    fig_lead(out, M)
    fig_operating_points(out, M)
    fig_subgroups(out, M)
    fig_robustness(out, M)
    fig_agp(out, ctx["agp"])
    fig_forecast(out, ctx["te"], ctx["abs_pred"], ctx["lo"], ctx["hi"])


def rebuild_from_metrics(out="results/figures"):
    """Regenerate every figure that depends only on results/metrics.json (no retraining needed)."""
    import json
    root = Path(__file__).resolve().parent.parent
    M = json.loads((root / "results" / "metrics.json").read_text())
    out = root / out
    _style()
    imp = pd.Series(M["top_features"]).sort_values(ascending=False)
    gimp = pd.Series(M["group_importance"]).sort_values(ascending=False)
    for fn in (fig_ablation, fig_cold, fig_horizon, fig_families, fig_calibration, fig_lead, fig_operating_points, fig_subgroups, fig_robustness):
        fn(out, M)
    fig_importance(out, imp, gimp)


if __name__ == "__main__":
    rebuild_from_metrics()
    print("figures rebuilt from metrics.json")
