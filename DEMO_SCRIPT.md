# Demo video script (target: 20 minutes)

The challenge page says 15-20 minutes while the guidelines say "minimum 20 minutes". Aim for **20:00 to 21:00** so you satisfy both. Record screen and voice, upload to YouTube as **Unlisted**, paste the link in `README.md` (section 10).

**Before recording**
```bash
pip install -r requirements.txt
pytest -q                          # shows the tests pass
streamlit run app/dashboard.py     # leave running
```
Open: the slide deck (`docs/DiaTwin_Presentation.pptx`), a terminal in the repo, the dashboard in the browser (zoom 110%).

| Time | Segment | What to show and say |
|---|---|---|
| 0:00-1:30 | Intro | Team, college, one-line pitch. Slide 1. |
| 1:30-3:00 | Problem | 101M Indians with diabetes; care is reactive; the next two hours are invisible. Slides 2-3. |
| 3:00-4:00 | Brief compliance | Walk the table on slide 4: each challenge requirement and where it is met. |
| 4:00-7:00 | Data | Slides 5-6. Terminal: `head data/ehr.csv`. Open `src/simulate_sensors.py` and walk through the physiology in the docstring (meal kernel, drug effect, exercise, dropouts). Show the AGP and the consensus-metric table; state the caveat that the cohort is more benign than real life. |
| 7:00-9:00 | Architecture | Slides 7-8. Open `src/features.py`: the 43 features in four groups, NaN handling. |
| 9:00-11:30 | Models and training | Slide 9. Explain learned vs mechanistic vs explanation layers, delta targets, modality-dropout training. In the terminal start `python -m src.train` (about 9 minutes: do not wait, show the 12 stages in the log and talk over it, or show `results/metrics.json` from a previous run). |
| 11:30-15:30 | Results | Slides 10-20 in order: ablation with CIs (modest but significant gain), cold start (-20%), horizons and model families, calibration, **operating points and alert fatigue (be candid about the 58% false-alert share at the tuned threshold)**, hypoglycaemia (no AUC advantage, say so), subgroups, robustness (plain vs modality-dropout), distribution shift and CV, explainability. |
| 15:30-19:30 | Live dashboard | See the click-path below. |
| 19:30-20:45 | Honesty and close | Slides 22-25: safety, where it fits (add your Tracxn startups), limits, roadmap. End on the README. |

## Live dashboard click-path (4 minutes)

1. **Twin view, hyperglycaemia.** Choose the highest-HbA1c patient. Drag the replay slider until the tile flips WATCH to ALERT. Tick "Reveal what actually happened" to show the forecast against reality.
2. **Why this forecast?** Point at the group bars: CGM trend and EHR push the forecast; meals and wearables contribute less.
3. **Hypoglycaemia moment.** Select **P305** (insulin) and set the replay slider to **359**: the hypo tile shows ALERT while glucose is still ~159 mg/dL; tick "Reveal" and show it fall to about 55 mg/dL within two hours. A second example: **P305 at 1577** (glucose 152, falls to ~54), or **P208 at 431**.
4. **What-if planner.** Planned meal 90 g, then compare eat vs half vs walk after vs skip.
5. **Cohort board.** Show the 12 patients ranked by risk at the current time.
6. **Data-quality simulator.** Tick "Wearables offline", then "Meal diary not used", then "CGM dropout": the twin keeps forecasting and shows a warning each time. Tie it to slide 17 (plain model would have failed).
7. **Model performance tab.** Open Alerts (operating-point table) and Subgroups.

Tips: speak to the honesty points (synthetic data, small hypoglycaemia sample, noisy default alerts), because they build credibility. Run `pytest -q` on camera once. End with the README open.
