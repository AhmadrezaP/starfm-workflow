"""Build every quantitative table, figure and in-text number for the v2 manuscript.

Inputs (copied from the Swan campaign, see ../data/README.md):
  data/DATE_PLANS/final_main_v4/results_merged.csv   one row per planned prediction
  data/DATE_PLANS/final_main_v4/failures.csv         cause of each excluded prediction
  data/TESTS/<test_id>/run_manifest.json             per-run provenance
  data/BASELINES/final_main/results/<test_id>.json   STARFM and temporal baselines on common pixels
  data/BASELINES/examples/<test_id>.npz              example crops

Outputs: tables/*.tex, figures/*.pdf|png, numbers.tex (LaTeX macros used in the text),
and summary.json. Run from SAT_manuscript_v2:  python analysis/build_results.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES, FIGURES = ROOT / "tables", ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BANDS = ["B1", "B2", "B3", "B4", "B5", "B7"]
BAND_NAMES = {"B1": "Coastal", "B2": "Blue", "B3": "Green", "B4": "Red", "B5": "NIR", "B7": "SWIR2"}
REGIONS = {
    (29, 32): ("NE-KS", "Nebraska--Kansas, USA"),
    (43, 34): ("CA", "San Joaquin Valley, California, USA"),
    (143, 42): ("UP", "Indo-Gangetic Plain, Uttar Pradesh, India"),
}
REGION_ORDER = ["NE-KS", "CA", "UP"]
# Validated categorical slots 1-3 (reference palette, all-pairs check passed).
COLORS = {"starfm": "#2a78d6", "linear_interpolation": "#eb6834", "persistence_nearest": "#1baf7a"}
LABELS = {"starfm": "STARFM", "linear_interpolation": "Linear interpolation", "persistence_nearest": "Nearest-date persistence"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f19c9b", "#b8312f"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})

numbers: dict[str, str] = {}


def number(name: str, value, fmt: str = "{}") -> None:
    numbers[name] = fmt.format(value)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def write_table(name: str, body: str) -> None:
    (TABLES / f"{name}.tex").write_text(body)


def clipped_scatter(ax, x, y, floor, **style):
    """Scatter with values below ``floor`` drawn as hollow triangles at the floor."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    low = y < floor
    ax.scatter(x[~low], y[~low], **style)
    ax.scatter(x[low], np.full(low.sum(), floor), marker="v", facecolor="none",
               edgecolor=style.get("color"), s=style.get("s", 6) * 2.5, lw=0.7)


def num(value: float, digits: int = 4, plus: bool = False) -> str:
    """Format a number for LaTeX with a typographic minus sign."""
    text = f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"
    return text.replace("-", "$-$")


def iqr(series: pd.Series, digits: int = 4) -> str:
    q1, q2, q3 = series.quantile([0.25, 0.5, 0.75])
    return f"{num(q2, digits)} [{num(q1, digits)}, {num(q3, digits)}]"


# ----------------------------------------------------------------------------- data
plan = pd.read_csv(DATA / "DATE_PLANS/final_main_v4/results_merged.csv")
plan["region"] = [REGIONS[(p, r)][0] for p, r in zip(plan.path, plan.row)]
failures = pd.read_csv(DATA / "DATE_PLANS/final_main_v4/failures.csv")
ok = plan[plan.status == "completed"].copy()
for column in ["rmse_surface_reflectance", "mae_surface_reflectance", "r2_surface_reflectance",
               "mean_difference_predicted_minus_actual", "valid_comparison_percent",
               "coregistration_x_shift_pixels", "coregistration_y_shift_pixels", "coregistration_quality_percent"]:
    ok[column] = pd.to_numeric(ok[column])
ok = ok.rename(columns={"rmse_surface_reflectance": "rmse", "mae_surface_reflectance": "mae",
                        "r2_surface_reflectance": "r2", "mean_difference_predicted_minus_actual": "bias"})
ok["shift_at_bound"] = (ok.coregistration_x_shift_pixels.abs() >= 20) | (ok.coregistration_y_shift_pixels.abs() >= 20)
ok["case"] = ok.region + " " + ok.prediction_date

# Development (tuning) campaign: its prediction cases are not independent tests.
development = pd.read_csv(DATA / "development/COREGISTRATION_FILES/OUTPUTS/DATE_PLANS/multiregion_v1_retry1/results.csv")
development_cases = {(p, r, d) for p, r, d in zip(development.path, development.row, development.prediction_date)}
development_dates = development_cases | {(p, r, x) for p, r, pairs in zip(development.path, development.row, development.pair_dates)
                                         for x in pairs.split(";")}
plan["tuning_case"] = [(p, r, d) in development_cases for p, r, d in zip(plan.path, plan.row, plan.prediction_date)]
ok["tuning_case"] = [(p, r, d) in development_cases for p, r, d in zip(ok.path, ok.row, ok.prediction_date)]
ok["shares_tuning_date"] = [any((p, r, x) in development_dates for x in [d, *pairs.split(";")])
                            for p, r, d, pairs in zip(ok.path, ok.row, ok.prediction_date, ok.pair_dates)]

manifests = {Path(p).parent.name: json.loads(Path(p).read_text()) for p in glob.glob(str(DATA / "TESTS/*/run_manifest.json"))}
metrics_files = glob.glob(str(DATA / "TESTS/*/VALIDATION/**/heldout_validation_metrics.json"), recursive=True)
means = {Path(p).parts[-8]: json.loads(Path(p).read_text()) for p in metrics_files}
ok["actual_mean"] = ok.test_id.map(lambda t: means[t]["actual_mean_surface_reflectance"])
ok["predicted_mean"] = ok.test_id.map(lambda t: means[t]["predicted_mean_surface_reflectance"])

# ----------------------------------------------------------------------- inventory
number("plannedPredictions", len(plan))
number("plannedCases", plan.groupby(["region", "prediction_date"]).ngroups)
number("completedPredictions", len(ok))
number("completedPercent", 100 * len(ok) / len(plan), "{:.1f}")
number("excludedPredictions", len(plan) - len(ok))
number("casesWithResults", ok.groupby(["region", "prediction_date"]).ngroups)
number("sixPercentPredictions", int((ok.campaign == "final_main_c6").sum()))
for cause, label in [("missing_coarse_file", "Missing"), ("coregistration_insufficient_overlap", "Coreg"),
                     ("no_valid_validation_pixels", "NoValid")]:
    number(f"excluded{label}", int((failures.cause == cause).sum()))
years = ok.prediction_date.str[:4].astype(int)
number("firstYear", years.min())
number("lastYear", years.max())

land_cover = {}
for manifest in manifests.values():
    for day, entry in (manifest.get("land_cover_years") or {}).items():
        land_cover[(manifest["wrs2_path"], manifest["wrs2_row"], day)] = entry
number("landCoverDates", len(land_cover))
number("landCoverSubstituted", sum(e["acquisition_year"] != e["mcd12q1_year"] for e in land_cover.values()))

pair_gap = {}
for manifest in manifests.values():
    pairs = sorted(pd.to_datetime(manifest["selected_pair_dates"]))
    target = pd.Timestamp(manifest["prediction_date"])
    pair_gap[manifest["test_id"]] = ((target - pairs[0]).days, (pairs[1] - target).days, (pairs[1] - pairs[0]).days)
ok["pair_span_days"] = ok.test_id.map(lambda t: pair_gap[t][2])
cloud = [p["scene_cloud_percent"] for m in manifests.values() for p in m["landsat_pairs"]]
target_cloud = [m["heldout_landsat_reference"]["scene_cloud_percent"] for m in manifests.values()]
number("maxPairCloud", max(cloud), "{:.2f}")
number("medianTargetCloud", float(np.median(target_cloud)), "{:.2f}")

rows = []
for key in REGION_ORDER:
    part, done = plan[plan.region == key], ok[ok.region == key]
    (path, row), (_, name) = next(((k, v) for k, v in REGIONS.items() if v[0] == key))
    sensors = "VIIRS (all bands)" if key == "UP" else "MODIS (B2--B7); VIIRS (B1)"
    name = {"NE-KS": "Nebraska--Kansas", "CA": "San Joaquin Valley", "UP": "Uttar Pradesh"}[key]
    rows.append(f"{name} & {path:03d}/{row:03d} & {sensors} & {part.prediction_date.nunique()} & "
                f"{part.prediction_date.min()[:4]}--{part.prediction_date.max()[:4]} & {len(part)} & {len(done)} \\\\")
write_table("inventory", "\\begin{tabularx}{\\linewidth}{L l P{2.6cm} r l r r}\n\\toprule\n"
            "Region & Path/row & Coarse input & Cases & Years & Planned & Scored\\\\\n\\midrule\n"
            + "\n".join(rows) + f"\n\\midrule\nTotal & & & {plan.groupby(['region','prediction_date']).ngroups} & & {len(plan)} & {len(ok)}\\\\\n"
            "\\bottomrule\n\\end{tabularx}\n")

cause_rows = []
for cause, text in [("missing_coarse_file", "Required coarse daily or BRDF granule absent from the archive"),
                    ("coregistration_insufficient_overlap", "Too few valid overlapping pixels for shift estimation"),
                    ("no_valid_validation_pixels", "No clear target pixels shared with the prediction")]:
    sub = failures[failures.cause == cause]
    if len(sub):
        granules = {}
        for item in sorted({d for d in sub.detail.dropna() if d}):
            product, day = item.split()
            granules.setdefault(product, []).append(day)
        detail = "; ".join(f"{product}: {', '.join(days)}" for product, days in sorted(granules.items())) or "--"
        cause_rows.append(f"{text} & {len(sub)} & {sub.prediction_date.nunique()} & {detail}\\\\")
write_table("exclusions", "\\begin{tabularx}{\\linewidth}{L r r P{4.8cm}}\n\\toprule\n"
            "Reason & $n$ & Dates & Missing granules (acquisition date)\\\\\n\\midrule\n" + "\n".join(cause_rows)
            + "\n\\bottomrule\n\\end{tabularx}\n")

# ------------------------------------------------------------------ development
dev = development[development.status == "completed"].copy()
dev["rmse"] = pd.to_numeric(dev.rmse_surface_reflectance)
dev_key = ["path", "row", "prediction_date", "band"]
number("devCases", development.groupby(["path", "row", "prediction_date"]).ngroups)
number("devRuns", len(development))
number("devCompleted", len(dev))
number("tuningCasesInFinal", int(plan[plan.tuning_case].groupby(["region", "prediction_date"]).ngroups))
number("independentCases", int(plan[~plan.tuning_case].groupby(["region", "prediction_date"]).ngroups))
number("independentScored", int((~ok.tuning_case).sum()))
number("tuningScored", int(ok.tuning_case.sum()))
number("sharesTuningDate", int((ok.shares_tuning_date & ~ok.tuning_case).sum()))
dev_rows = []
for family, column, chosen, label in [("distance", "distance", 950, "Search distance (m)"),
                                       ("uncertainty", "uncertainty", 35, "Uncertainty"),
                                       ("coregistration", "method", "moving-window", "Registration")]:
    table = dev[dev.family == family].pivot_table(index=dev_key, columns=column, values="rmse")
    complete = table.dropna()
    wins = complete.idxmin(axis=1).value_counts()
    attempted = development[development.family == family].groupby(column).size()
    finished = dev[dev.family == family].groupby(column).size()
    for option in table.columns:
        name = {"arosics": "AROSICS", "moving-window": "Moving window", "none": "None"}.get(option, option)
        dev_rows.append(f"{label if option == table.columns[0] else ''} & {name}{' (selected)' if option == chosen else ''} & "
                        f"{finished.get(option, 0)}/{attempted.get(option, 0)} & {complete[option].median():.4f} & "
                        f"{int(wins.get(option, 0))}/{len(complete)}\\\\")
    dev_rows.append("\\midrule")
    number(f"dev{family.capitalize()}Complete", len(complete))
    number(f"dev{family.capitalize()}ChosenWins", int(wins.get(chosen, 0)))
d = dev[dev.family == "distance"]
number("devFarDistanceCompleted", int((d.distance == 1150).sum()))
number("devFarDistanceAttempted", int((development[development.family == "distance"].distance == 1150).sum()))
number("devFarDistanceWins", int(dev[dev.family == "distance"].pivot_table(index=dev_key, columns="distance", values="rmse").dropna().idxmin(axis=1).eq(1150).sum()))
sensors = dev[dev.family == "sensors"].pivot_table(index=dev_key, columns="sensor", values="rmse").dropna()
for (p, r), (key, _) in REGIONS.items():
    part = sensors[[i[0] == p and i[1] == r for i in sensors.index]]
    preferred = "VIIRS" if key == "UP" else "MODIS"
    other = "MODIS" if preferred == "VIIRS" else "VIIRS"
    dev_rows.append(f"{'Coarse sensor' if key == REGION_ORDER[0] else ''} & {key}: {preferred} (selected) vs {other} & "
                    f"{len(part)} pairs & {part[preferred].median():.4f} / {part[other].median():.4f} & "
                    f"{int((part[preferred] < part[other]).sum())}/{len(part)}\\\\")
    number(f"devSensorWins{key.replace('-', '')}", int((part[preferred] < part[other]).sum()))
    number(f"devSensorPairs{key.replace('-', '')}", len(part))
write_table("development", "\\begin{tabularx}{\\linewidth}{P{2.0cm} L r r r}\n\\toprule\n"
            "Factor & Option & Completed & Median RMSE & Lowest RMSE\\\\\n\\midrule\n"
            + "\n".join(dev_rows[:-1] if dev_rows[-1] == "\\midrule" else dev_rows) + "\n\\bottomrule\n\\end{tabularx}\n")

# ------------------------------------------------------------------ STARFM accuracy
# Relative error of the scene mean, as reported by Wang et al. (2014): 100 (predicted - actual) / actual.
ok["rel_error"] = 100 * (ok.predicted_mean - ok.actual_mean) / ok.actual_mean
independent = ok[~ok.tuning_case]
number("indepMedianRMSE", independent.rmse.median(), "{:.4f}")
number("tuningMedianRMSE", ok[ok.tuning_case].rmse.median(), "{:.4f}")
number("medianRMSE", ok.rmse.median(), "{:.4f}")
number("medianMAE", ok.mae.median(), "{:.4f}")
number("medianAbsRelError", ok.rel_error.abs().median(), "{:.1f}")
number("largeRelError", int((ok.rel_error.abs() > 10).sum()))
number("largeRelErrorPercent", 100 * (ok.rel_error.abs() > 10).mean(), "{:.0f}")
number("largeRelErrorUP", int((ok[ok.region == "UP"].rel_error.abs() > 10).sum()))
number("medianCoverage", ok.valid_comparison_percent.median(), "{:.1f}")
number("shiftAtBound", int(ok.shift_at_bound.sum()))
number("medianAbsShift", float(np.median(np.hypot(ok.coregistration_x_shift_pixels, ok.coregistration_y_shift_pixels))), "{:.1f}")

rows = []
for band in BANDS:
    b = ok[ok.band == band]
    rows.append(f"{band} ({BAND_NAMES[band]}) & {len(b)} & {iqr(b.rmse)} & {b.mae.median():.4f} & "
                f"{num(b.bias.median(), 4, plus=True)} & {iqr(b.rel_error, 1)} & {(b.rel_error.abs() > 10).sum()} \\\\")
write_table("starfm_by_band", "\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}lrlrrlr@{}}\n\\toprule\n"
            "Band & $n$ & RMSE median [IQR] & MAE & Bias & Relative error (\\%) & $>$10\\%\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for key in REGION_ORDER:
    for sensor in ["MODIS", "VIIRS"]:
        g = ok[(ok.region == key) & (ok.sensor == sensor)]
        if not len(g):
            continue
        bands = "B1" if (sensor == "VIIRS" and key != "UP") else ("B1--B7" if key == "UP" else "B2--B7")
        rows.append(f"{key} & {sensor} & {bands} & {len(g)} & {g.prediction_date.nunique()} & {iqr(g.rmse)} & "
                    f"{iqr(g.rel_error, 1)} & {(g.rel_error.abs() > 10).sum()}\\\\")
write_table("starfm_by_region", "\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}llllrllr@{}}\n\\toprule\n"
            "Region & Sensor & Bands & $n$ & Dates & RMSE median [IQR] & Relative error (\\%) & $>$10\\%\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# -------------------------------------------------------------------------- baselines
baseline_files = sorted(glob.glob(str(DATA / "BASELINES/final_main/results/*.json")))
records = []
for path in baseline_files:
    d = json.loads(Path(path).read_text())
    row = dict(test_id=d["test_id"], band=d["band"], prediction_date=d["prediction_date"],
               region=REGIONS[(d["wrs2_path"], d["wrs2_row"])][0], sensor=d["coarse_sensor"],
               common_fraction=d["common_fraction_of_saved_mask"], gap_early=d["days_from_earlier_pair"],
               gap_late=d["days_to_later_pair"])
    for method, values in d["common_mask"].items():
        for metric in ("rmse", "mae", "bias", "r2"):
            row[f"{method}_{metric}"] = values[metric]
    records.append(row)
base = pd.DataFrame(records)
number("baselineRuns", len(base))
if len(base):
    base["span"] = base.gap_early + base.gap_late
    base["delta_linear"] = base.starfm_rmse - base.linear_interpolation_rmse
    base["delta_persistence"] = base.starfm_rmse - base.persistence_nearest_rmse
    base["case"] = base.region + " " + base.prediction_date
    base = base.merge(ok[["test_id", "tuning_case"]], on="test_id", how="left")
    indep = base[~base.tuning_case]
    for method, label in [("linear_interpolation", "Linear"), ("persistence_nearest", "Persistence")]:
        number(f"indepStarfmBeats{label}", int((indep.starfm_rmse < indep[f"{method}_rmse"]).sum()))
        number(f"indepStarfmBeats{label}Percent", 100 * (indep.starfm_rmse < indep[f"{method}_rmse"]).mean(), "{:.0f}")
    number("indepBaselineRuns", len(indep))
    number("medianCommonFraction", 100 * base.common_fraction.median(), "{:.0f}")
    for method, label in [("linear_interpolation", "Linear"), ("persistence_nearest", "Persistence")]:
        wins = (base.starfm_rmse < base[f"{method}_rmse"])
        number(f"starfmBeats{label}", int(wins.sum()))
        number(f"starfmBeats{label}Percent", 100 * wins.mean(), "{:.0f}")
        number(f"median{label}RMSE", base[f"{method}_rmse"].median(), "{:.4f}")
        relative = 100 * (base.starfm_rmse - base[f"{method}_rmse"]) / base[f"{method}_rmse"]
        number(f"medianRelDiff{label}", relative.median(), "{:+.0f}")
        case_wins = base.assign(w=wins).groupby("case").w.mean()
        number(f"casesStarfmMajority{label}", int((case_wins > 0.5).sum()))
    number("baselineCases", base.case.nunique())
    number("medianStarfmCommonRMSE", base.starfm_rmse.median(), "{:.4f}")
    relative_by_span = (100 * base.delta_linear / base.linear_interpolation_rmse).groupby(base.span)
    medians, shares = relative_by_span.median(), relative_by_span.apply(lambda v: (v > 0).mean())
    number("spanShortest", int(medians.index.min()))
    number("spanShortestMedian", medians.iloc[0], "{:.0f}")
    number("spanBestDays", int(medians.idxmin()))
    number("spanBestMedian", medians.min(), "{:.0f}")
    number("spanMinShareInterpBetter", 100 * shares.min(), "{:.0f}")

    rows = []
    for band in BANDS:
        b = base[base.band == band]
        rows.append(f"{band} & {len(b)} & {b.starfm_rmse.median():.4f} & {b.linear_interpolation_rmse.median():.4f} & "
                    f"{b.persistence_nearest_rmse.median():.4f} & {(b.starfm_rmse < b.linear_interpolation_rmse).sum()} & "
                    f"{(b.starfm_rmse < b.persistence_nearest_rmse).sum()}\\\\")
    for key in REGION_ORDER:
        b = base[base.region == key]
        rows.append(("\\midrule\n" if key == REGION_ORDER[0] else "") +
                    f"{key} & {len(b)} & {b.starfm_rmse.median():.4f} & {b.linear_interpolation_rmse.median():.4f} & "
                    f"{b.persistence_nearest_rmse.median():.4f} & {(b.starfm_rmse < b.linear_interpolation_rmse).sum()} & "
                    f"{(b.starfm_rmse < b.persistence_nearest_rmse).sum()}\\\\")
    rows.append(f"\\midrule\nAll & {len(base)} & {base.starfm_rmse.median():.4f} & {base.linear_interpolation_rmse.median():.4f} & "
                f"{base.persistence_nearest_rmse.median():.4f} & {(base.starfm_rmse < base.linear_interpolation_rmse).sum()} & "
                f"{(base.starfm_rmse < base.persistence_nearest_rmse).sum()}\\\\")
    write_table("baselines", "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                " & & \\multicolumn{3}{c}{Median RMSE} & \\multicolumn{2}{c}{STARFM lower RMSE than}\\\\\n"
                "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
                "Group & $n$ & STARFM & Linear & Persistence & Linear & Persistence\\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # Figure: STARFM vs linear interpolation, one point per prediction.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.38})
    ax = axes[0]
    limit = max(base.starfm_rmse.max(), base.linear_interpolation_rmse.max()) * 1.05
    ax.plot([0, limit], [0, limit], color="#c3c2b7", lw=1, zorder=0)
    markers = {"NE-KS": "o", "CA": "s", "UP": "^"}
    for key in REGION_ORDER:
        r = base[base.region == key]
        ax.scatter(r.linear_interpolation_rmse, r.starfm_rmse, s=14, marker=markers[key], facecolor="none",
                   edgecolor=COLORS["starfm"], lw=0.8, label=key)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Linear interpolation RMSE"); ax.set_ylabel("STARFM RMSE")
    ax.legend(title="Region", frameon=False, loc="lower right")
    ax.set_title("(a) Paired RMSE on common pixels", loc="left")
    ax = axes[1]
    spans = sorted(base.span.unique())
    data = [100 * (base[base.span == s].delta_linear / base[base.span == s].linear_interpolation_rmse) for s in spans]
    ax.axhline(0, color="#c3c2b7", lw=1)
    parts = ax.boxplot(data, positions=range(len(spans)), widths=0.55, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2))
    for patch in parts["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=COLORS["starfm"], lw=0.8)
    for i, d in enumerate(data):
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(d))
        values = np.asarray(d, dtype=float)
        high = values > 600
        ax.scatter((i + jitter)[~high], values[~high], s=5, color=COLORS["starfm"], alpha=0.6, lw=0)
        ax.scatter((i + jitter)[high], np.full(high.sum(), 600), marker="^", facecolor="none",
                   edgecolor=COLORS["starfm"], s=14, lw=0.7)
    ax.set_ylim(-110, 630)
    number("relDiffAboveSixHundred", int(sum((np.asarray(d) > 600).sum() for d in data)))
    ax.set_xticks(range(len(spans)), [f"{s}\n(n={len(d)})" for s, d in zip(spans, data)])
    ax.set_xlabel("Days between the two Landsat pairs"); ax.set_ylabel("STARFM RMSE relative to\nlinear interpolation (%)")
    ax.set_title("(b) Relative difference by pair span", loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    save(fig, "baselines")

# ------------------------------------------------- comparison with Wang et al. (2014)
# Whole-scene RMSE from Wang et al. (2014), Table III, WRS-2 143/042, Landsat 7 ETM+ with MODIS.
# Their ETM+ bands 1, 2, 3, 4 and 7 correspond to our B2, B3, B4, B5 and B7.
WANG = {"B2": (0.016, 0.015, 0.032), "B3": (0.021, 0.017, 0.023), "B4": (0.028, 0.023, 0.031),
        "B5": (0.033, 0.025, 0.055), "B7": (0.051, 0.043, 0.065)}
# Relative error of the scene mean (%), same table and band order.
WANG_RELATIVE = {"B2": (3.3, -8.6, -39.1), "B3": (7.3, -5.18, -13.8), "B4": (3.1, -4.7, -5.3),
                 "B5": (2.7, -1.3, 9.2), "B7": (4.0, -3.5, 13.4)}
up = ok[ok.region == "UP"]
rmse_rows, rel_rows, below_best, total = [], [], 0, 0
for band, values in WANG.items():
    b = up[up.band == band]
    best = min(values)
    below_best += int((b.rmse <= best).sum()); total += len(b)
    rmse_rows.append(f"{band} ({BAND_NAMES[band]}) & {values[0]:.3f} & {values[1]:.3f} & {values[2]:.3f} & "
                     f"{b.rmse.median():.3f} & [{b.rmse.quantile(0.25):.3f}, {b.rmse.quantile(0.75):.3f}]\\\\")
    rel = WANG_RELATIVE[band]
    rel_rows.append(f"{band} ({BAND_NAMES[band]}) & {num(rel[0], 1, plus=True)} & {num(rel[1], 1, plus=True)} & "
                    f"{num(rel[2], 1, plus=True)} & {num(b.rel_error.median(), 1, plus=True)} & "
                    f"[{num(b.rel_error.quantile(0.25), 1, plus=True)}, {num(b.rel_error.quantile(0.75), 1, plus=True)}]\\\\")
write_table("wang_comparison", "\\begin{tabular}{lrrrrl}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Wang et al. (2014)} & \\multicolumn{2}{c}{This study ($n=18$ per band)}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "Band & Test I & Test II & Test III & Median & IQR\\\\\n\\midrule\n"
            "\\multicolumn{6}{l}{\\textit{RMSE (reflectance)}}\\\\\n" + "\n".join(rmse_rows) +
            "\n\\midrule\n\\multicolumn{6}{l}{\\textit{Relative error of the scene mean (\\%)}}\\\\\n" + "\n".join(rel_rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
up5 = up[up.band.isin(list(WANG))]
wang_abs_rel = [abs(v) for values in WANG_RELATIVE.values() for v in values]
number("upMedianAbsRelError", up5.rel_error.abs().median(), "{:.1f}")
number("upAbsRelErrorWithinWangWorst", int((up5.rel_error.abs() <= max(wang_abs_rel)).sum()))
number("wangMaxAbsRelError", max(wang_abs_rel), "{:.1f}")
number("wangMedianAbsRelError", float(np.median(wang_abs_rel)), "{:.1f}")
number("upLargeRelError", int((up5.rel_error.abs() > 10).sum()))
number("wangLargeRelError", sum(v > 10 for v in wang_abs_rel))
number("upPredictions", len(up))
number("upDates", up.prediction_date.nunique())
number("wangComparablePredictions", total)
number("upAtOrBelowWangBest", below_best)
number("upAtOrBelowWangBestPercent", 100 * below_best / total, "{:.0f}")
number("upMedianRMSEFiveBands", up[up.band.isin(list(WANG))].rmse.median(), "{:.3f}")
number("wangMedianRMSE", float(np.median([v for values in WANG.values() for v in values])), "{:.3f}")

# ------------------------------------------------------------------- figure: accuracy
REL_CAP = 40
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, column, label in [(axes[0], "rmse", "RMSE (reflectance)"), (axes[1], "rel_error", "Relative error of scene mean (%)")]:
    data = [ok[ok.band == b][column] for b in BANDS]
    shown = [d.clip(upper=REL_CAP) if column == "rel_error" else d for d in data]
    parts = ax.boxplot(shown, positions=range(6), widths=0.55, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2))
    for patch in parts["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=COLORS["starfm"], lw=0.8)
    for i, d in enumerate(data):
        x = i + np.random.default_rng(10 + i).uniform(-0.18, 0.18, len(d))
        values = np.asarray(d, dtype=float)
        high = values > REL_CAP if column == "rel_error" else np.zeros(len(values), bool)
        ax.scatter(x[~high], values[~high], s=5, color=COLORS["starfm"], alpha=0.55, lw=0)
        ax.scatter(x[high], np.full(high.sum(), REL_CAP), marker="^", facecolor="none",
                   edgecolor=COLORS["starfm"], s=14, lw=0.7)
    ax.set_xticks(range(6), [f"{b}\n{BAND_NAMES[b]}" for b in BANDS])
    ax.set_ylabel(label)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
axes[1].axhline(0, color="#c3c2b7", lw=1)
axes[1].set_ylim(-35, REL_CAP + 4)
number("relErrorAboveCap", int((ok.rel_error > REL_CAP).sum()))
number("relErrorCap", REL_CAP)
axes[0].set_title("(a) RMSE by band", loc="left"); axes[1].set_title("(b) Relative error by band", loc="left")
save(fig, "accuracy_by_band")

# ------------------------------------------------------- figure: failure diagnostics
# STARFM RMSE as a multiple of linear-interpolation RMSE on common pixels; comparable across bands.
ok = ok.merge(base[["test_id", "starfm_rmse", "linear_interpolation_rmse"]], on="test_id", how="left")
ok["ratio"] = ok.starfm_rmse / ok.linear_interpolation_rmse
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
shown = ok.rel_error.clip(upper=80)
ax.scatter(shown, ok.ratio, s=7, color=COLORS["starfm"], alpha=0.7, lw=0)
ax.axvline(0, color="#c3c2b7", lw=1); ax.axhline(1, color="#c3c2b7", lw=1)
ax.set_yscale("log"); ax.set_xlim(-30, 85)
ax.set_xlabel("Relative error of scene mean (%)"); ax.set_ylabel("STARFM RMSE / interpolation RMSE")
number("biasAboveEighty", int((ok.rel_error > 80).sum()))
ax.set_title("(a) Scene-level radiometric offset", loc="left")
ax = axes[1]
groups = [ok[~ok.shift_at_bound].ratio, ok[ok.shift_at_bound].ratio]
parts = ax.boxplot(groups, positions=[0, 1], widths=0.5, patch_artist=True,
                   showfliers=False, medianprops=dict(color=INK, lw=1.2))
for patch in parts["boxes"]:
    patch.set(facecolor="#cde2fb", edgecolor=COLORS["starfm"], lw=0.8)
for i, g in enumerate(groups):
    ax.scatter(i + np.random.default_rng(30 + i).uniform(-0.15, 0.15, len(g)), g, s=5,
               color=COLORS["starfm"], alpha=0.55, lw=0)
ax.axhline(1, color="#c3c2b7", lw=1)
ax.set_yscale("log")
ax.set_xticks([0, 1], [f"Inside search range\n(n={len(groups[0])})", f"At $\\pm$20-pixel limit\n(n={len(groups[1])})"])
ax.set_ylabel("STARFM RMSE / interpolation RMSE")
ax.yaxis.grid(True, color=GRID, lw=0.5)
ax.set_title("(b) Registration shift at the search limit", loc="left")
save(fig, "diagnostics")
large = ok.rel_error.abs() > 10
poor = ok.ratio >= 2
number("biasCorrelationSpearman", ok.rel_error.abs().corr(ok.ratio, method="spearman"), "{:.2f}")
number("absRelBiasAboveTen", int(large.sum()))
number("largeOffsetPositive", int((large & (ok.rel_error > 0)).sum()))
number("poorPredictions", int(poor.sum()))
number("poorPredictionsPercent", 100 * poor.mean(), "{:.0f}")
number("poorWithLargeOffset", int((poor & large).sum()))
number("poorPositiveOffset", int((poor & (ok.rel_error > 0)).sum()))
number("poorUP", int((poor & (ok.region == "UP")).sum()))
number("ratioAtBound", ok[ok.shift_at_bound].ratio.median(), "{:.2f}")
number("ratioNotBound", ok[~ok.shift_at_bound].ratio.median(), "{:.2f}")
number("medianRatio", ok.ratio.median(), "{:.2f}")
worst = ok.nlargest(10, "ratio")
number("worstTenLargePositiveOffset", int((worst.rel_error > 10).sum()))
number("worstTenDates", worst.case.nunique())
number("worstTenMinOffset", worst.rel_error.min(), "{:.0f}")
number("worstTenMaxOffset", worst.rel_error.max(), "{:.0f}")
number("worstTenVisible", int(worst.band.isin(["B1", "B2", "B3"]).sum()))

# ------------------------------------------------------------- figure: examples
examples = sorted(glob.glob(str(DATA / "BASELINES/examples/*.npz")))
if examples:
    fig, axes = plt.subplots(len(examples), 5, figsize=(7.2, 1.55 * len(examples) + 0.4), squeeze=False)
    for i, path in enumerate(examples):
        e = np.load(path, allow_pickle=False)
        test_id = Path(path).stem
        region = ok.loc[ok.test_id == test_id, "region"].iloc[0]
        stats = ok.loc[ok.test_id == test_id].iloc[0]
        actual = np.where(e["mask"], e["actual"], np.nan)
        valid = e["actual"][e["mask"]]
        lo, hi = np.nanpercentile(valid, [2, 98])
        panels = [("Earlier pair", e["early"]), ("Held-out target", actual), ("STARFM", e["starfm"]),
                  ("Linear interpolation", e["linear"])]
        for j, (title, values) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(values, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if i == 0:
                ax.set_title(title)
        diff = np.where(e["mask"], e["starfm"] - e["actual"], np.nan)
        bound = float(np.nanpercentile(np.abs(diff), 98))
        image = axes[i, 4].imshow(diff, cmap=DIVERGING, vmin=-bound, vmax=bound, interpolation="nearest")
        if i == 0:
            axes[i, 4].set_title("STARFM $-$ target")
        cbar = fig.colorbar(image, ax=axes[i, 4], fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6)
        d1, target, d2 = [str(x) for x in e["dates"]]
        axes[i, 0].set_ylabel(f"{region} {str(e['band'])}\n{target}\nRMSE {stats.rmse:.3f}", fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    save(fig, "examples")

# ------------------------------------------------------------------------ outputs
(ROOT / "numbers.tex").write_text("% Generated by analysis/build_results.py; do not edit.\n" + "".join(
    f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(numbers.items())))
(ROOT / "analysis/summary.json").write_text(json.dumps(numbers, indent=2) + "\n")
ok.to_csv(ROOT / "analysis/starfm_runs.csv", index=False)
if len(base):
    base.to_csv(ROOT / "analysis/baseline_runs.csv", index=False)
print(json.dumps(numbers, indent=2))
