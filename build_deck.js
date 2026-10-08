// Builds docs/DiaTwin_Presentation.pptx from results/metrics.json   (node docs/build_deck.js)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(ROOT, "results/figures", n);
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics.json"), "utf8"));
const OUT = path.join(__dirname, "DiaTwin_Presentation.pptx");

const THEME = {
  name: "DiaTwin", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2433", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF3F7", accent1: "0E9F8E", accent2: "E4572E", accent3: "F2A541",
            accent4: "5C8EA8", accent5: "9DB4C0", accent6: "8A94A6", hlink: "0E9F8E", folHlink: "5C8EA8" },
};
const HEX = { navy: "14213D", teal: "0E9F8E", coral: "E4572E", amber: "F2A541", soft: "EEF3F7", mid: "5C8EA8", grey: "8A94A6" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "DiaTwin - Digital Twin for Type 2 Diabetes";
pres.author = "Team DiaTwin, IMT Ghaziabad";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
pres.defineSlideMaster({ title: "DARK", background: { color: HEX.navy }, objects: [] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" }, objects: [],
  slideNumber: { x: 9.2, y: 5.25, color: HEX.grey, fontFace: "Calibri", fontSize: 10 }, margin: [0.5, 0.5, 0.5, 0.5] });

// ---------------------------------------------------------------- helpers
const fmt = (v, d = 1) => Number(v).toFixed(d);
const R = M.regression, CL = M.classification, HY = M.hypoglycaemia, CS = M.cold_start, B = M.bootstrap, D = M.dataset, G = D.glycemic;
const AL = M.alerts.fusion, OP = M.operating_points, tuned = OP.find((o) => o.tuned), RB = M.robustness, SH = M.distribution_shift, CV = M.cross_validation;
const gain = B.mae_fusion_vs_cgm.fusion.gain_vs_cgm_only;
const cutP = Math.round((1 - R.fusion.mae / R.persistence.mae) * 100);
const cutC = Math.round((1 - CS.cold_fusion.mae / CS.cold_cgm.mae) * 100);

async function icon(Comp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: String(size) }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
function title(s, text, color = C.text2) {
  s.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.75, fontFace: "Cambria", fontSize: 32, bold: true, color, margin: 0, valign: "middle", isTextBox: true, objectName: "Title" });
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.12, objectName: "Card" });
}
async function circle(s, Comp, x, y, d = 0.6, bg = HEX.teal) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg, width: 0 }, objectName: "IconCircle" });
  s.addImage({ data: await icon(Comp), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: "Icon" });
}
function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, ...o });
}
function bullets(s, items, x, y, w, h, o = {}) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontFace: "Calibri", fontSize: 14, color: C.text1, paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true, ...o });
}
function stat(s, big, label, sub, x, y, w = 2.1, color = HEX.teal) {
  s.addText(big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: 34, bold: true, color, margin: 0, valign: "middle", isTextBox: true });
  s.addText(label, { x, y: y + 0.78, w, h: 0.32, fontFace: "Calibri", fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x, y: y + 1.1, w, h: 0.5, fontFace: "Calibri", fontSize: 11, color: HEX.grey, margin: 0, valign: "top", isTextBox: true });
}
async function img(s, file, x, y, maxW, maxH, alt) {
  const m = await sharp(file).metadata();
  const r = Math.min(maxW / m.width, maxH / m.height);
  const w = m.width * r, h = m.height * r;
  s.addImage({ path: file, x: x + (maxW - w) / 2, y: y + (maxH - h) / 2, w, h, altText: alt, objectName: alt });
}
const sub = (s, t) => text(s, t, 0.5, 5.0, 8.6, 0.3, { fontSize: 11, color: HEX.grey });
const chartOpts = (names, vals, colors, ttl) => ({
  x: 0.5, y: 1.2, w: 5.7, h: 3.8, barDir: "col", chartColors: colors, showTitle: true, title: ttl, titleFontSize: 14, titleColor: "14213D", titleFontFace: "+mn-lt",
  showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelColor: "14213D", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: "14213D", catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelColor: "8A94A6", valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, valAxisMinVal: 0,
});

(async () => {
  let s;
  const content = (sec) => pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });

  // ============ 1. Title ============
  pres.addSection({ title: "Opening" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  await circle(s, fa.FaHeartbeat, 0.7, 0.9, 0.9, HEX.teal);
  s.addText("DiaTwin", { x: 0.7, y: 2.0, w: 8.6, h: 1.0, fontFace: "Cambria", fontSize: 60, bold: true, color: "FFFFFF", margin: 0, isTextBox: true, objectName: "Title" });
  text(s, "A digital twin for Type 2 Diabetes that forecasts glucose, warns before spikes and lows, explains why, and simulates what-if choices",
    0.7, 3.05, 7.8, 1.1, { fontSize: 20, color: "CFE9E6" });
  text(s, "Digital Twin Challenge 2026  |  Happiest Health  |  Team leader: Shivansh Porwal, IMT Ghaziabad", 0.7, 4.75, 8.6, 0.4, { fontSize: 12, color: "9DB4C0" });
  s.addNotes("Introduce the team and the one-line pitch. DiaTwin fuses a patient's EHR with live wearable and CGM data, forecasts glucose 30, 60 and 120 minutes ahead, warns before a spike or a low, explains each forecast and lets the doctor simulate choices.");

  // ============ 2. Problem ============
  pres.addSection({ title: "Problem and solution" });
  s = content("Problem and solution");
  title(s, "Diabetes care is reactive");
  [["101M", "Indians living with diabetes"], ["136M", "more with prediabetes"], ["11.4%", "of the population has diabetes"]].forEach(([big, lab], i) => {
    const y = 1.3 + i * 1.2;
    s.addText(big, { x: 0.5, y, w: 2.6, h: 0.9, fontFace: "Cambria", fontSize: 48, bold: true, color: HEX.teal, margin: 0, valign: "middle", isTextBox: true, objectName: `Stat${i}` });
    text(s, lab, 3.1, y, 2.2, 0.9, { valign: "middle" });
  });
  card(s, 5.6, 1.3, 3.9, 3.5);
  text(s, "What clinicians see today", 5.85, 1.45, 3.4, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2 });
  bullets(s, ["Lab values such as HbA1c, reviewed every few months", "Occasional glucose readings", "No view of what the next two hours hold", "Spikes, and lows on insulin or sulfonylureas, go unseen"], 5.85, 1.95, 3.4, 2.7);
  sub(s, "Source: ICMR-INDIAB study, The Lancet Diabetes & Endocrinology, 2023");
  s.addNotes("About 101 million people with diabetes and 136 million with prediabetes in India (ICMR-INDIAB, Lancet Diabetes & Endocrinology, 2023). Care is mostly retrospective. A digital twin makes it proactive.");

  // ============ 3. Solution ============
  s = content("Problem and solution");
  title(s, "Meet DiaTwin");
  const four = [[fa.FaLayerGroup, "Fuse", "Merge the static EHR with CGM, wearable and meal-diary streams for each patient.", HEX.teal],
    [fa.FaChartLine, "Forecast", "Glucose at +30, +60 and +120 minutes with an 80% uncertainty band.", HEX.mid],
    [fa.FaBell, "Warn", "Hyper and hypo alerts, with a why-this-forecast panel for the doctor.", HEX.amber],
    [fa.FaFlask, "Simulate", "What if I eat, halve the meal, walk or skip? See the curve first.", HEX.coral]];
  for (let i = 0; i < 4; i++) {
    const x = 0.5 + i * 2.3;
    card(s, x, 1.3, 2.15, 3.2);
    await circle(s, four[i][0], x + 0.2, 1.5, 0.65, four[i][3]);
    text(s, four[i][1], x + 0.2, 2.3, 1.8, 0.45, { fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2 });
    text(s, four[i][2], x + 0.2, 2.8, 1.8, 1.6, { fontSize: 13 });
  }
  text(s, "Scope: one condition, Type 2 Diabetes, as the challenge requires. A proof of concept on synthetic data.", 0.5, 4.75, 9, 0.35, { italic: true, color: HEX.mid });
  s.addNotes("Four verbs: fuse, forecast, warn and explain, simulate. Type 2 Diabetes was chosen because it is prevalent in India, glucose is measurable continuously and the two-hour window is actionable.");

  // ============ 4. Brief compliance ============
  s = content("Problem and solution");
  title(s, "Built to the challenge brief");
  const rows = [["Challenge requirement", "How DiaTwin meets it"],
    ["Fuse static EHR with dynamic real-time data", "Synthetic EHR + simulated CGM, heart rate, HRV, steps, sleep and meal diary: 43 fused features"],
    ["Algorithmic model that predicts an adverse event", "2-hour glucose forecast, spike-onset and hypoglycaemia alerts, with uncertainty"],
    ["Conceptual UI dashboard for a doctor", "Working Streamlit app: twin view, cohort board, what-if planner, explanations"],
    ["Anonymised, open-source or synthetic data only", "100% synthetic; Synthea-compatible importer; DPDP Act and HIPAA safe"],
    ["Public GitHub repo with full README", "README with every required item, MIT licence, tests, CI, Docker"],
    ["Demo video, architecture diagram, presentation", "20-minute script, 2-page architecture PDF, this deck"]];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "1B2433", fill: { color: i === 0 ? HEX.navy : (i % 2 ? "FFFFFF" : HEX.soft) }, fontFace: "Calibri", fontSize: 12, valign: "middle" } }))),
    { x: 0.5, y: 1.25, w: 9, colW: [3.6, 5.4], rowH: 0.52, border: { type: "solid", color: "DDE3EA", pt: 0.5 }, margin: [0.04, 0.1, 0.04, 0.1] });
  s.addNotes("This slide maps each line of the challenge brief to what we built, so the jury can tick them off.");

  // ============ 5. Data ============
  pres.addSection({ title: "Build" });
  s = content("Build");
  title(s, "Two data streams, one patient");
  card(s, 0.5, 1.25, 4.4, 2.3);
  await circle(s, fa.FaUserMd, 0.7, 1.4, 0.55, HEX.mid);
  text(s, "Static: simulated EHR", 1.4, 1.4, 3.4, 0.55, { fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, valign: "middle" });
  bullets(s, ["Demographics, BMI, disease duration", "HbA1c, fasting glucose, eGFR, LDL", "TCF7L2 genetic risk marker", "Medication, comorbidities, lifestyle"], 0.7, 2.05, 4.0, 1.45);
  card(s, 5.1, 1.25, 4.4, 2.3);
  await circle(s, fa.FaHeartbeat, 5.3, 1.4, 0.55, HEX.teal);
  text(s, "Dynamic: wearables", 6.0, 1.4, 3.4, 0.55, { fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, valign: "middle" });
  bullets(s, ["CGM glucose every 5 min, with dropouts", "Heart rate, HRV, steps, sleep stages", "Meal diary, only 85% logged", "Medication effects: dosing, skipped meals"], 5.3, 2.05, 4.0, 1.45);
  [[String(D.patients), "virtual patients"], [String(D.days_per_patient), "days each"], [`${fmt(D.sensor_rows / 1e6, 1)}M`, "sensor readings"], [String(D.n_features_fusion), "fused features"]].forEach(([big, lab], i) => {
    const x = 0.5 + i * 2.28;
    s.addText(big, { x, y: 3.75, w: 2.1, h: 0.6, fontFace: "Cambria", fontSize: 32, bold: true, color: HEX.teal, margin: 0, isTextBox: true });
    text(s, lab, x, 4.35, 2.1, 0.35);
  });
  sub(s, "Synthetic only, as the sandbox rules require: DPDP Act and HIPAA safe by construction.");
  s.addNotes("Static stream: a synthetic EHR with a Synthea-compatible schema and importer. Dynamic stream: simulated CGM, heart rate, HRV, steps, sleep and a meal diary. Imperfections are modelled on purpose: CGM dropouts and unlogged meals.");

  // ============ 6. Credibility ============
  s = content("Build");
  title(s, "Is the synthetic data credible?");
  await img(s, FIG("agp_cohort.png"), 0.4, 1.15, 5.3, 2.6, "AGP");
  const grows = [["Metric", "Cohort", "Consensus"],
    ["Time in range 70-180", `${fmt(G.tir_70_180_pct, 0)}%`, "> 70%"], ["Time above 180", `${fmt(G.tar_over_180_pct, 0)}%`, "< 25%"],
    ["Time below 70", `${fmt(G.tbr_under_70_pct, 2)}%`, "< 4%"], ["GMI", `${fmt(G.gmi_pct, 1)}%`, "-"], ["CV", `${fmt(G.cv_pct, 0)}%`, "<= 36%"]];
  s.addTable(grows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "1B2433", fill: { color: i === 0 ? HEX.navy : (i % 2 ? "FFFFFF" : HEX.soft) }, fontFace: "Calibri", fontSize: 12, valign: "middle" } }))),
    { x: 5.9, y: 1.25, w: 3.6, colW: [1.7, 0.95, 0.95], rowH: 0.4, border: { type: "solid", color: "DDE3EA", pt: 0.5 } });
  card(s, 0.5, 3.95, 9, 1.0, HEX.soft);
  text(s, "Honest caveat: the cohort is better controlled and less variable than typical real-world Indian Type 2 Diabetes, so absolute errors are optimistic. What transfers: model ranking, cold-start effect, robustness and the evaluation method.",
    0.7, 4.05, 8.6, 0.8, { fontSize: 13, valign: "middle", color: C.text2 });
  s.addNotes("The AGP is the clinical standard way of showing CGM data: median and percentile bands by time of day. The three meal peaks are visible. Cohort metrics sit in the consensus ranges. We state plainly that it is more benign than real life.");

  // ============ 7. Architecture ============
  s = content("Build");
  title(s, "System architecture");
  await img(s, path.join(__dirname, "architecture_diagram.png"), 0.5, 1.15, 9, 3.95, "Architecture diagram");
  s.addNotes("Left to right: two streams, the fusion layer, the twin core (a learned layer and a mechanistic layer), and four clinician-facing outputs: alerts, explanations, the what-if planner and the cohort board with data quality.");

  // ============ 8. Pipeline ============
  s = content("Build");
  title(s, "ML pipeline and evaluation");
  await img(s, path.join(__dirname, "architecture_pipeline.png"), 0.5, 1.15, 9, 3.95, "ML pipeline diagram");
  s.addNotes("Reproducible with one command. Twelve evaluation tests, all patient-level so no patient appears in more than one split.");

  // ============ 9. Model ============
  s = content("Build");
  title(s, "How the twin predicts");
  [["Learned layer", HEX.teal, "Gradient-boosted trees on 43 fused features: three forecast horizons, 10-90% band, spike and hypo classifiers. Modality-dropout training."],
   ["Mechanistic layer", HEX.amber, "Personal insulin sensitivity from the EHR drives meal and exercise kernels for what-if scenarios."],
   ["Explanation layer", HEX.mid, "Neutralise each data group to see how much it moves the forecast."]].forEach(([t, c, d], i) => {
    const y = 1.25 + i * 1.28;
    card(s, 0.5, y, 4.4, 1.15);
    text(s, t, 0.7, y + 0.08, 4.0, 0.35, { fontFace: "Cambria", fontSize: 16, bold: true, color: c });
    text(s, d, 0.7, y + 0.45, 4.0, 0.7, { fontSize: 12 });
  });
  stat(s, fmt(R.fusion.mae), "MAE in mg/dL at 2 h", `persistence ${fmt(R.persistence.mae)}`, 5.2, 1.25);
  stat(s, fmt(CL.fusion.roc_auc, 2), "ROC-AUC, spike alert", `calibration ECE ${fmt(M.calibration.spike.ece, 3)}`, 7.5, 1.25);
  stat(s, `${Math.round(R.fusion.within15mgdl_pct)}%`, "within 15 mg/dL", "of true glucose", 5.2, 3.1);
  stat(s, `${Math.round(R.fusion_band_80pct_coverage)}%`, "band coverage", "nominal 80%", 7.5, 3.1);
  s.addNotes(`Metrics are on ${D.train_val_test_patients[2]} held-out patients. Spike onset means glucose is currently at or below 180 and crosses 180 within 2 hours. The band is meant to cover 80% and covers ${fmt(R.fusion_band_80pct_coverage, 0)}%.`);

  // ============ 10. Ablation ============
  pres.addSection({ title: "Results" });
  s = content("Results");
  title(s, "Does fusion help?");
  const keys = ["persistence", "cgm_only", "cgm+wearables", "cgm+ehr", "fusion"];
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: ["Persistence", "CGM only", "CGM + wearables", "CGM + EHR", "Full fusion"], values: keys.map((k) => Number(fmt(R[k].mae))) }],
    chartOpts(0, 0, ["8A94A6", "9DB4C0", "5C8EA8", "5C8EA8", "0E9F8E"], "2-hour forecast error (MAE, mg/dL)"));
  stat(s, `-${cutP}%`, "error vs persistence", `95% CI ${fmt(B.mae.fusion.ci95[0])}-${fmt(B.mae.fusion.ci95[1])} mg/dL`, 6.6, 1.25, 2.9);
  stat(s, `${fmt(gain.mean, 2)}`, "mg/dL gain over CGM only", `paired 95% CI ${fmt(gain.ci95[0], 2)} to ${fmt(gain.ci95[1], 2)}`, 6.6, 2.95, 2.9);
  s.addNotes(`Patient-level bootstrap. Fusion beats CGM only by ${fmt(gain.mean, 2)} mg/dL with a CI that excludes zero: a real but modest gain, because glucose history already reveals the patient's baseline. The next slide shows where the EHR matters most.`);

  // ============ 11. Cold start ============
  s = content("Results");
  title(s, "The cold-start advantage");
  const ck = ["cold_cgm", "cold_cgm+wearables", "cold_cgm+ehr", "cold_fusion"];
  s.addChart(pres.charts.BAR, [{ name: "MAE", labels: ["CGM (<1 h)", "CGM + wearables", "CGM + EHR", "Full fusion"], values: ck.map((k) => Number(fmt(CS[k].mae))) }],
    chartOpts(0, 0, ["9DB4C0", "5C8EA8", "5C8EA8", "0E9F8E"], "New patient: forecast error (MAE, mg/dL)"));
  stat(s, `-${cutC}%`, "error for a new patient", `${fmt(CS.cold_cgm.mae)} to ${fmt(CS.cold_fusion.mae)} mg/dL`, 6.6, 1.25, 2.9);
  text(s, "With under an hour of CGM data, the EHR supplies the baseline the sensor cannot yet show. That is what a twin is for: a head start on day one.", 6.6, 2.95, 2.9, 1.9);
  s.addNotes("Cold start: only the last 30 minutes of glucose, no 24-hour statistics and no sleep history. The EHR closes most of the gap.");

  // ============ 12. Horizons + families ============
  s = content("Results");
  title(s, "Horizons and model families");
  await img(s, FIG("horizon_mae.png"), 0.4, 1.2, 4.6, 3.2, "Horizon chart");
  await img(s, FIG("model_families.png"), 5.0, 1.2, 4.6, 3.2, "Model families chart");
  const h = M.horizons;
  const fam = M.model_families, bestFam = Math.min(...Object.values(fam).map((x) => x.mae)), oursMae = fam["HistGradientBoosting (ours)"].mae;
  const famTxt = oursMae <= bestFam + 1e-9 ? `Gradient boosting is the most accurate of four model families (extra-trees ${fmt(fam["Extra-trees (80)"].mae)}, MLP ${fmt(fam["MLP (64-32)"].mae)}); ridge regression is far behind (${fmt(fam["Ridge regression"].mae)}).`
    : `Gradient boosting is within ${fmt(oursMae - bestFam, 2)} mg/dL of the best of four model families; ridge regression is far behind.`;
  text(s, `Error at +30 / +60 / +120 min: ${fmt(h["30"].fusion.mae)} / ${fmt(h["60"].fusion.mae)} / ${fmt(h["120"].fusion.mae)} mg/dL. ${famTxt}`, 0.5, 4.5, 9, 0.6, { fontSize: 13 });
  s.addNotes("We did not just pick one model. Ridge, extra-trees, an MLP and gradient boosting were compared on identical features. Boosting was chosen for native missing-value handling, speed and quantile support, not because it is uniquely best.");

  // ============ 13. Alerts ============
  s = content("Results");
  title(s, "Alerts you can trust");
  await img(s, FIG("calibration_spike.png"), 0.4, 1.2, 3.6, 3.4, "Calibration");
  stat(s, fmt(CL.fusion.roc_auc, 3), "spike-onset ROC-AUC", `95% CI ${fmt(B.spike_auc.fusion.ci95[0], 3)}-${fmt(B.spike_auc.fusion.ci95[1], 3)}`, 4.3, 1.25, 2.4);
  stat(s, fmt(M.calibration.spike.brier, 3), "Brier score", `ECE ${fmt(M.calibration.spike.ece, 3)}`, 6.9, 1.25, 2.4);
  stat(s, `+${fmt(B.spike_auc_gain_fusion_vs_cgm.mean, 3)}`, "AUC gain vs CGM only", `CI ${fmt(B.spike_auc_gain_fusion_vs_cgm.ci95[0], 3)} to ${fmt(B.spike_auc_gain_fusion_vs_cgm.ci95[1], 3)}`, 4.3, 3.05, 2.4);
  stat(s, `${Math.round(CL.fusion.recall * 100)}% / ${Math.round(CL.fusion.precision * 100)}%`, "recall / precision", "at the tuned threshold", 6.9, 3.05, 2.4);
  s.addNotes("The predicted probabilities match observed frequencies closely, which matters if a clinician is going to read '70% risk' as meaning 70%.");

  // ============ 14. Operating points ============
  s = content("Results");
  title(s, "Alert fatigue is a design choice");
  await img(s, FIG("operating_points.png"), 0.4, 1.15, 4.6, 3.5, "Operating points chart");
  const orows = [["Threshold", "Alerts / day", "False share", "Flagged >=30 min"]].concat(OP.map((o) => [`${fmt(o.threshold, 2)}${o.tuned ? " *" : ""}`, fmt(o.alert_episodes_per_patient_day), `${fmt(o.false_alert_share_pct, 0)}%`, `${fmt(o.detected_with_30min_lead_pct, 0)}%`]));
  s.addTable(orows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : "1B2433", fill: { color: i === 0 ? HEX.navy : (i % 2 ? "FFFFFF" : HEX.soft) }, fontFace: "Calibri", fontSize: 11, valign: "middle", align: "center" } }))),
    { x: 5.2, y: 1.25, w: 4.3, colW: [0.9, 1.0, 1.0, 1.4], rowH: 0.36, border: { type: "solid", color: "DDE3EA", pt: 0.5 } });
  text(s, `At the tuned threshold (*), ${fmt(tuned.false_alert_share_pct, 0)}% of alert episodes are not followed by a spike. Raising the threshold cuts false alerts but misses more excursions. The clinician chooses the trade-off.`, 5.2, 3.7, 4.3, 1.2, { fontSize: 12.5 });
  s.addNotes("This is the honest weak point. At the default threshold the alert is sensitive but noisy. We expose the whole trade-off table rather than hide it. A deployed system would add snooze logic and per-patient thresholds.");

  // ============ 15. Hypo + lead time ============
  s = content("Results");
  title(s, "Early warning and hypoglycaemia");
  await img(s, FIG("alert_lead_time.png"), 0.4, 1.15, 4.7, 3.2, "Lead time histogram");
  card(s, 5.3, 1.25, 4.2, 3.4);
  text(s, "Hypoglycaemia: a rare event", 5.5, 1.38, 3.8, 0.4, { fontFace: "Cambria", fontSize: 17, bold: true, color: C.text2 });
  bullets(s, [`${D.hypo_positive_rows_test} positive rows from ${D.hypo_positive_patients_test} patients (${D.insulin_patients_test} on insulin)`,
    `ROC-AUC ${fmt(HY.fusion.roc_auc, 3)} vs CGM only ${fmt(HY.cgm_only.roc_auc, 3)}: no AUC advantage from the EHR`,
    `Recall ${Math.round(HY.fusion.recall * 100)}% at precision ${Math.round(HY.fusion.precision * 100)}%: a screening aid, not a diagnosis`], 5.5, 1.85, 3.8, 2.7, { fontSize: 13 });
  text(s, `Spike warnings come a median ${fmt(AL.median_lead_min, 0)} min ahead (capped at 120) and flag ${fmt(AL.detected_pct, 1)}% of excursions.`, 0.5, 4.5, 4.6, 0.5, { fontSize: 12.5 });
  s.addNotes("Lead time is capped by the two-hour look-back. For hypoglycaemia we report the result as it is: with few events from few insulin users, the EHR does not improve AUC over CGM alone, so this is a proof of concept.");

  // ============ 16. Subgroups ============
  s = content("Results");
  title(s, "Does it work for everyone?");
  await img(s, FIG("subgroup_mae.png"), 0.4, 1.1, 5.6, 3.95, "Subgroup chart");
  const big = M.subgroups.filter((r) => r.patients >= 10);
  const worst = big.reduce((a, b) => (b.mae_fusion > a.mae_fusion ? b : a)), best = big.reduce((a, b) => (b.mae_fusion < a.mae_fusion ? b : a));
  const minImp = Math.min(...big.map((r) => Math.round((1 - r.mae_fusion / r.mae_persistence) * 100)));
  stat(s, `${fmt(best.mae_fusion)}-${fmt(worst.mae_fusion)}`, "MAE range, mg/dL", `${best.dimension} ${best.group} to ${worst.dimension} ${worst.group}`, 6.3, 1.25, 3.2);
  stat(s, `>= ${minImp}%`, "better than persistence", "in every subgroup of 10+ patients", 6.3, 3.0, 3.2);
  s.addNotes("Five dimensions: HbA1c band, medication, age, sex and activity. Performance is fairly even; the weakest groups are still far better than the baseline. Small groups such as the six insulin users are flagged in the repo, not over-read.");

  // ============ 17. Robustness ============
  s = content("Results");
  title(s, "Robust when inputs break");
  await img(s, FIG("robustness_mae.png"), 0.4, 1.1, 5.9, 3.9, "Robustness chart");
  stat(s, fmt(RB.wearables_offline.fusion_plain_mae), "plain model, wearables off", `worse than CGM only (${fmt(RB.wearables_offline.cgm_only_mae)})`, 6.5, 1.25, 3.0, HEX.coral);
  stat(s, fmt(RB.wearables_offline.fusion_mae), "modality-dropout model", "same scenario, same data", 6.5, 3.0, 3.0);
  s.addNotes("Real devices fail. A plain fusion model collapses when the wearables go offline. Training with random masking of wearables and the meal diary keeps the error at the clean level. The dashboard has a data-quality simulator to show this live.");

  // ============ 18. Shift + CV ============
  s = content("Results");
  title(s, "Holds up beyond the test set");
  card(s, 0.5, 1.25, 4.4, 3.4);
  text(s, "Distribution shift", 0.7, 1.35, 4.0, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2 });
  text(s, `Unseen cohort of ${SH.cohort.patients}: older (${fmt(SH.cohort.mean_age, 0)} vs ${fmt(SH.cohort.train_mean_age, 0)} y), heavier (BMI ${fmt(SH.cohort.mean_bmi)} vs ${fmt(SH.cohort.train_mean_bmi)}), longer disease (${fmt(SH.cohort.mean_duration_y)} vs ${fmt(SH.cohort.train_mean_duration_y)} y).`, 0.7, 1.8, 4.0, 1.0, { fontSize: 12.5 });
  stat(s, fmt(SH.fusion_mae), "fusion MAE", `CGM only ${fmt(SH.cgm_only_mae)}, persistence ${fmt(SH.persistence_mae)}`, 0.7, 2.7, 4.0);
  card(s, 5.1, 1.25, 4.4, 3.4);
  text(s, "5-fold patient-level CV", 5.3, 1.35, 4.0, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2 });
  text(s, "Whole patients are held out in every fold.", 5.3, 1.8, 4.0, 0.5, { fontSize: 12.5 });
  stat(s, `${fmt(CV.fusion.mean, 2)} +/- ${fmt(CV.fusion.sd, 2)}`, "fusion MAE", `CGM only ${fmt(CV.cgm_only.mean, 2)} +/- ${fmt(CV.cgm_only.sd, 2)}`, 5.3, 2.05, 4.0);
  stat(s, `${fmt(CV.cold_fusion.mean, 1)}`, "cold-start fusion MAE", `CGM-only cold ${fmt(CV.cold_cgm.mean, 1)}`, 5.3, 3.25, 4.0);
  s.addNotes("Two more ways to check we are not fooling ourselves: an independent cohort with a different age, weight and disease-duration mix, and five-fold cross-validation grouped by patient. Rankings hold; error rises under shift because that population is harder.");

  // ============ 19. Explainability ============
  s = content("Results");
  title(s, "Why this forecast?");
  await img(s, FIG("group_importance.png"), 0.4, 1.15, 5.2, 2.4, "Group importance");
  const gi = M.group_importance;
  text(s, "Globally, glucose history dominates, the EHR is second, meals third and wearables add least in this simulation, partly because exercise already shows up in the CGM trace.", 0.5, 3.65, 5.0, 1.0, { fontSize: 12.5 });
  card(s, 5.8, 1.25, 3.7, 3.4);
  text(s, "In the dashboard", 6.0, 1.35, 3.3, 0.4, { fontFace: "Cambria", fontSize: 17, bold: true, color: C.text2 });
  bullets(s, ["Per-patient bars show how each data group pushes this forecast up or down", "Red raises glucose, green lowers it", "Stream-health warnings when data is missing"], 6.0, 1.85, 3.3, 2.7, { fontSize: 13 });
  s.addNotes(`Permutation importance by group: CGM ${fmt(gi["Glucose history & trend (CGM)"])}, EHR ${fmt(gi["EHR profile"])}, meals ${fmt(gi["Meals (diary)"])}, wearables ${fmt(gi["Activity, HR, HRV, sleep (wearables)"])} mg/dL. The per-patient explanation neutralises a group to its training median and reports how the forecast moves.`);

  // ============ 20. Forecast vs reality ============
  s = content("Results");
  title(s, "Forecast vs reality");
  await img(s, FIG("forecast_example.png"), 0.5, 1.15, 8.0, 3.55, "Forecast vs actual");
  stat(s, `${Math.round(R.fusion_band_80pct_coverage)}%`, "inside the 80% band", "", 8.45, 1.4, 1.2);
  text(s, "Held-out virtual patient. Each forecast was made two hours earlier. The sharpest peaks are underestimated; the band widens to show uncertainty.", 0.5, 4.75, 8.4, 0.5, { fontSize: 11.5, color: HEX.grey });
  s.addNotes("The twin anticipates the shape and timing of meal excursions. Sharp peaks are underestimated, which is a known limitation of any regression-to-the-mean forecaster.");

  // ============ 21. Dashboard ============
  s = content("Results");
  title(s, "The doctor's view");
  const dash = [[fa.FaHeartbeat, "Twin view", "Forecasts, hyper and hypo tiles, why-this-forecast, clinical summary.", HEX.teal],
    [fa.FaUsers, "Cohort board", "Every patient ranked by risk right now.", HEX.mid],
    [fa.FaFlask, "What-if planner", "Eat, halve, walk after or skip: the next three hours.", HEX.coral],
    [fa.FaDatabase, "Data fusion", "EHR profile, wearable streams and the fused feature vector.", HEX.amber],
    [fa.FaPlug, "Data-quality simulator", "Switch off wearables, diary or CGM and watch it degrade safely.", HEX.mid],
    [fa.FaChartLine, "Model performance", "Every evaluation in this deck, interactive.", HEX.teal]];
  for (let i = 0; i < 6; i++) {
    const x = 0.5 + (i % 3) * 3.05, y = 1.25 + Math.floor(i / 3) * 1.85;
    card(s, x, y, 2.9, 1.7);
    await circle(s, dash[i][0], x + 0.2, y + 0.2, 0.55, dash[i][3]);
    text(s, dash[i][1], x + 0.9, y + 0.2, 1.9, 0.55, { fontFace: "Cambria", fontSize: 14, bold: true, color: C.text2, valign: "middle" });
    text(s, dash[i][2], x + 0.2, y + 0.9, 2.55, 0.75, { fontSize: 12 });
  }
  sub(s, "Streamlit app in the repository: streamlit run app/dashboard.py   |   Docker image included");
  s.addNotes("Switch to the live dashboard here. Pick patient P305 (insulin), set the replay slider to 359 and show the hypoglycaemia alert about two hours before glucose falls to the mid-50s. Then flip the data-quality toggles.");

  // ============ 22. Safety ============
  pres.addSection({ title: "Context and close" });
  s = content("Context and close");
  title(s, "Safe, private, accountable");
  const three = [[fa.FaUserShield, "Human in the loop", ["Decision support only: it never doses", "Uncertainty is always shown", "Missing streams are flagged, not hidden"], HEX.teal],
    [fa.FaShieldAlt, "Privacy by design", ["No real patient data", "Synthetic EHR and sensors", "DPDP Act and HIPAA safe"], HEX.mid],
    [fa.FaBalanceScale, "Fair and regulated", ["Performance by 5 subgroup types", "Alert-fatigue trade-off exposed", "Software-as-medical-device review before any clinical use"], HEX.coral]];
  for (let i = 0; i < 3; i++) {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.25, 2.85, 3.6);
    await circle(s, three[i][0], x + 0.25, 1.45, 0.65, three[i][3]);
    text(s, three[i][1], x + 0.25, 2.25, 2.4, 0.5, { fontFace: "Cambria", fontSize: 17, bold: true, color: C.text2 });
    bullets(s, three[i][2], x + 0.25, 2.85, 2.45, 1.95, { fontSize: 13 });
  }
  s.addNotes("The tool supports a clinician and never doses. All data is synthetic. A real deployment needs consent, data minimisation, in-region processing and a regulatory review as software that could influence treatment.");

  // ============ 23. Where it fits ============
  s = content("Context and close");
  title(s, "Where DiaTwin fits");
  card(s, 0.5, 1.25, 4.4, 2.6);
  text(s, "Common approaches today", 0.75, 1.4, 3.9, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.grey });
  bullets(s, ["CGM apps: show glucose history and trends", "Care programmes: coaching and logging", "Hospital analytics: periodic risk scores"], 0.75, 2.0, 3.9, 1.8, { fontSize: 16 });
  card(s, 5.1, 1.25, 4.4, 2.6, HEX.teal);
  text(s, "DiaTwin", 5.35, 1.4, 3.9, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: "FFFFFF" });
  bullets(s, ["Fuses EHR and wearables per patient", "Forecasts, warns and explains", "Simulates choices before they are made"], 5.35, 2.0, 3.9, 1.8, { fontSize: 16, color: "FFFFFF" });
  text(s, "The gap we close: personalised, forward-looking, explainable and simulatable.", 0.5, 4.1, 9, 0.5, { fontFace: "Cambria", fontSize: 20, bold: true, color: HEX.teal });
  sub(s, "Our assessment of common approaches, not an exhaustive market survey.");
  s.addNotes("Before recording, replace or extend the left column with specific startups from the ValleyNxt Ventures Tracxn screen: name, what they do, what they do not do. That is the team's strongest differentiator.");

  // ============ 24. Limits + roadmap ============
  s = content("Context and close");
  title(s, "Limits and roadmap");
  card(s, 0.5, 1.25, 4.4, 3.2);
  text(s, "What we do not claim", 0.7, 1.35, 4.0, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.coral });
  bullets(s, ["All results are on synthetic data: not clinical claims", "Simulated CGM is less variable than real life", "What-if layer shares the simulator's assumptions", `Hypoglycaemia: few events, ${D.hypo_positive_patients_test} test patients`, "Default alert threshold is noisy"], 0.7, 1.85, 4.0, 3.0, { fontSize: 12.5 });
  card(s, 5.1, 1.25, 4.4, 3.2);
  text(s, "Next steps", 5.3, 1.35, 4.0, 0.4, { fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.teal });
  bullets(s, ["Validate on real anonymised CGM and EHR data", "Learn the mechanistic parameters from data", "Add dose and medication timing; sequence models", "Clinician usability study, per-patient alert thresholds", "Drift monitoring and regulatory review"], 5.3, 1.85, 4.0, 3.0, { fontSize: 12.5 });
  s.addNotes("Be candid: absolute numbers are properties of our simulator. The transferable results are the ranking of models, the cold-start effect, the robustness to missing streams and the evaluation methodology.");

  // ============ 25. Close ============
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Context and close" });
  s.addText("Predict. Warn. Simulate.", { x: 0.7, y: 1.6, w: 8.6, h: 1.0, fontFace: "Cambria", fontSize: 42, bold: true, color: "FFFFFF", margin: 0, isTextBox: true, objectName: "Title" });
  text(s, "DiaTwin turns a patient's records and wearables into a virtual twin that a doctor can consult before the spike or the low, not after.", 0.7, 2.8, 7.6, 0.9, { fontSize: 18, color: "CFE9E6" });
  text(s, "Team DiaTwin  |  IMT Ghaziabad  |  Code, README, technical report and demo video in the public GitHub repository", 0.7, 4.75, 8.6, 0.4, { fontSize: 12, color: "9DB4C0" });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
