"""
Supplementary Figure 2: Association between lesion volume and non-lesional
white-matter R1 and R2 values in the RRMS cohort.

The script calculates subject-level lesion burden and non-lesional white-matter
summaries and generates the corresponding association analyses and figure.

Outputs:
- subject-level CSV with lesion volume and non-lesional WM R1/R2 mean/std
- two-panel association figure for lesion volume vs non-lesional WM R1/R2
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

try:
	from utils import paths
	from utils.paper_figure_helpers import *
	from utils.paper_meta import *
	from utils.paper_stats import *
	from utils.plotting import apply_figure_style, add_subplot_label, create_figure as make_figure
except ImportError as exc:
	raise ImportError(
		"Failed to import project modules. Ensure you are running from the project root."
	) from exc


# -----------------------------------------------------------------------------
# cfguration
# -----------------------------------------------------------------------------

cfg = yaml.safe_load(open(paths.default_config, "r", encoding="utf-8"))

METRICS = {
	"t1_pinn": {
		"label": "R1",
		"legend_label":  r"$\mathbf{R_1}$",
		"ylabel": r"$\mathbf{R_1}$ [1/s]",
		"color": "tab:orange",
	},
	"t2_pinn": {
		"label": "R2",
		"legend_label": r"$\mathbf{R_2}$",
		"ylabel": r"$\mathbf{R_2}$ [1/s]",
		"color": "tab:orange",
	},
}

FA_MASK = (cfg["fa_mask_lower"], cfg["fa_mask_upper"])
MS_SEG_ROOT = Path("/mnt/DATA/3t-ukt/source/MS_seg")

EXP_NAME = Path(__file__).stem
FIG_DIR = paths.root / "fig" / EXP_NAME
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = get_paper_figure_cache_dir(EXP_NAME)


# -----------------------------------------------------------------------------
# Cohort and lesion-volume loading
# -----------------------------------------------------------------------------


def _load_metadata() -> pd.DataFrame:
	metadata_fp = (
		paths.data
		/ "251217-b_get_gender_age_participants"
		/ "251217-b_participant_metadata.csv"
	)
	metadata = pd.read_csv(metadata_fp)
	metadata["subject_id"] = metadata["subject_id"].astype(str).str.strip()
	metadata["subject_name"] = metadata["subject_name"].astype(str).str.strip()
	metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")
	return metadata


def _build_rrms_subject_table(subject_map: dict[str, str]) -> pd.DataFrame:
	metadata = _load_metadata()
	hc_subject_ids = get_hc_baseline_subject_ids(subject_map)
	rrms_subject_ids = get_age_matched_rrms_subject_ids(subject_map, hc_subject_ids)

	rrms_meta = metadata[
		metadata["subject_id"].isin(rrms_subject_ids)
		& metadata["subject_name"].str.contains("MS", na=False)
	].copy()

	rrms_meta = rrms_meta.dropna(subset=["subject_id", "subject_name", "age"])
	rrms_meta = rrms_meta.sort_values(["subject_name", "subject_id"]).reset_index(drop=True)
	return rrms_meta


def _resolve_ms_seg_folder(subject_name: str) -> Path:
	candidates = [p for p in MS_SEG_ROOT.iterdir() if p.is_dir() and p.name.startswith(subject_name)]
	if not candidates:
		raise FileNotFoundError(f"No MS_seg folder found for {subject_name} under {MS_SEG_ROOT}")
	return sorted(candidates, key=lambda p: (len(p.name), p.name))[-1]


def _resolve_lesion_volume_yaml(subject_name: str) -> Path:
	seg_dir = _resolve_ms_seg_folder(subject_name)
	candidates = [
		p for p in seg_dir.glob("t2*_filt_*.yml")
		if "_unfilt_" not in p.name and "_order" not in p.name
	]
	if not candidates:
		candidates = [p for p in seg_dir.glob("**/*_filt_*.yml") if "_unfilt_" not in p.name]
		if not candidates:
			raise FileNotFoundError(f"No filtered lesion YAML found in {seg_dir}")
	return sorted(candidates, key=lambda p: (len(p.name), p.name))[-1]


@lru_cache(maxsize=None)
def load_lesion_volume_ml(subject_name: str) -> float:
	yml_fp = _resolve_lesion_volume_yaml(subject_name)
	with open(yml_fp, "r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle) or {}
	if "volume_in_ml" not in data:
		raise KeyError(f"volume_in_ml missing in {yml_fp}")
	return float(data["volume_in_ml"])


# -----------------------------------------------------------------------------
# non-lesional WM metrics
# -----------------------------------------------------------------------------


def build_mask(sid: str, exclude_lesions: bool = False) -> np.ndarray | None:
    """Build refined WM mask for a subject."""
    # Load R1/R2 WM limits from defaults.yaml (consistent with figure2)
    r1_limits = (cfg.get("r1_mask_wm_lower", np.nan),
                 cfg.get("r1_mask_wm_upper", np.nan))
    r2_limits = (cfg.get("r2_mask_wm_lower", np.nan),
                 cfg.get("r2_mask_wm_upper", np.nan))

    # Use the cerebral WM mask builder from utils.paper_figure_helpers
    return build_subject_cerebral_wm_mask(
        sid,
        r1_limits=r1_limits,
        r2_limits=r2_limits,
        exclude_lesions=exclude_lesions,
    )

def compute_nonlesional_wm_summary(subject_id: str) -> dict[str, float] | None:
	mask = build_mask(subject_id, exclude_lesions=True)
	if mask is None:
		return None

	lesion_mask = load_subject_lesion_mask(subject_id)
	if lesion_mask is not None and lesion_mask.shape == mask.shape:
		mask = mask & ~lesion_mask

	if not np.any(mask):
		return None

	out: dict[str, float] = {}
	for metric_key, info in METRICS.items():
		data = load_subject_metric(metric_key, subject_id)
		if data is None:
			out[f"{info['label'].lower()}_mean"] = np.nan
			out[f"{info['label'].lower()}_std"] = np.nan
			continue

		vals = np.asarray(data[mask], dtype=float)
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			out[f"{info['label'].lower()}_mean"] = np.nan
			out[f"{info['label'].lower()}_std"] = np.nan
			continue

		out[f"{info['label'].lower()}_mean"] = float(np.mean(vals))
		out[f"{info['label'].lower()}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

	return out


def build_subject_table(subject_map: dict[str, str]) -> pd.DataFrame:
	rrms_meta = _build_rrms_subject_table(subject_map)
	records: list[dict[str, object]] = []

	for row in rrms_meta.to_dict("records"):
		subject_id = str(row["subject_id"])
		subject_name = str(row["subject_name"])
		lesion_volume = load_lesion_volume_ml(subject_name)
		nawm = compute_nonlesional_wm_summary(subject_id)
		if nawm is None:
			continue

		records.append(
			{
				"subject_name": subject_name,
				"subject_id": subject_id,
				"age": float(row["age"]),
				"gender": str(row.get("gender", "")),
				"lesion_volume_ml": lesion_volume,
				"r1_mean": nawm["r1_mean"],
				"r1_std": nawm["r1_std"],
				"r2_mean": nawm["r2_mean"],
				"r2_std": nawm["r2_std"],
			}
		)

	df = pd.DataFrame(records)
	if not df.empty:
		df = df.sort_values(["lesion_volume_ml", "subject_name"]).reset_index(drop=True)
	return df


def save_subject_table(df: pd.DataFrame) -> Path | None:
	if df.empty:
		return None
	out_fp = CACHE_DIR / "rrms_nawm_lesion_volume_subject_values.csv"
	# sort df by MS id
	df = df.sort_values(["subject_name", "subject_id"]).reset_index(drop=True)
	df.to_csv(out_fp, index=False)
	print(f"Saved subject-level table: {out_fp}")
	return out_fp


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _fit_and_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
	valid = np.isfinite(x) & np.isfinite(y)
	x = x[valid]
	y = y[valid]
	if x.size < 2:
		return {
			"x": x,
			"y": y,
			"rho": np.nan,
			"p": np.nan,
			"slope": np.nan,
			"intercept": np.nan,
		}

	rho, p = stats.spearmanr(x, y)
	slope, intercept, *_ = stats.linregress(x, y)
	return {
		"x": x,
		"y": y,
		"rho": float(rho),
		"p": float(p),
		"slope": float(slope),
		"intercept": float(intercept),
	}


def _should_use_log_x(x: np.ndarray) -> bool:
	x = np.asarray(x, dtype=float)
	x = x[np.isfinite(x) & (x > 0)]
	if x.size < 4:
		return False
	ratio = float(np.max(x) / np.min(x))
	return ratio >= 5.0


def plot_association_panels(df: pd.DataFrame) -> plt.Figure:
	apply_figure_style()

	fig = make_figure(height=5.6, width=10.4)
	gs = GridSpec(1, 2, figure=fig, wspace=0.28)
	axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
	subplot_labels = get_subplot_labels(2)
	y_axis_lims = {
		"t1_pinn": (1.5, 1.85),
		"t2_pinn": (17, 21),
	}

	x = df["lesion_volume_ml"].to_numpy(dtype=float)
	use_log_x = _should_use_log_x(x)

	for ax, metric_key, label, sublabel in zip(
		axes,
		["t1_pinn", "t2_pinn"],
		["R1", "R2"],
		subplot_labels,
	):
		meta = METRICS[metric_key]
		y = df[f"{label.lower()}_mean"].to_numpy(dtype=float)
		# yerr = df[f"{label.lower()}_std"].to_numpy(dtype=float)

		stats_out = _fit_and_stats(x, y)
		finite_x = stats_out["x"]
		finite_y = stats_out["y"]

		ax.errorbar(
			finite_x,
			finite_y,
			# yerr=yerr[np.isfinite(x) & np.isfinite(y)],
			fmt="o",
			ms=7,
			color=meta["color"],
			ecolor=meta["color"],
			elinewidth=1.1,
			capsize=3,
			markeredgecolor="black",
			markeredgewidth=0.6,
			alpha=0.9,
			zorder=3,
		)

		if finite_x.size >= 2:
			x_line = np.linspace(float(np.min(finite_x)), float(np.max(finite_x)), 100)
			y_line = stats_out["intercept"] + stats_out["slope"] * x_line
			ax.plot(x_line, y_line, color=meta["color"], linewidth=2.0, zorder=2)

		if use_log_x:
			ax.set_xscale("log")

		ax.grid(True, linestyle="--", alpha=0.35)
		ax.set_xlabel("Total lesion volume (mL)", fontsize=13, fontweight="bold")
		ax.set_ylabel(meta["ylabel"], fontsize=13, fontweight="bold")
		# ax.set_title(meta["label"], fontsize=13, fontweight="bold")
		ax.tick_params(axis="both", which="major", labelsize=11)
		ax.set_ylim(*y_axis_lims[metric_key])

		if np.isfinite(stats_out["rho"]):
			p_text = f"{stats_out['p']:.3g}" if np.isfinite(stats_out["p"]) else "nan"
			ax.text(
				0.04,
				0.96,
				f"Spearman $\\rho$ = {stats_out['rho']:.2f}\n$p$ = {p_text}",
				transform=ax.transAxes,
				va="top",
				ha="left",
				fontsize=11,
				bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2.5),
			)

		add_subplot_label(ax, sublabel)

	# legend_handles = [
	# 	mlines.Line2D([], [], color=METRICS[m]["color"], marker="o", linestyle="None",
	# 				  markeredgecolor="black", markeredgewidth=0.6, markersize=7,
	# 				  label=METRICS[m]["legend_label"])
	# 	for m in ["t1_pinn", "t2_pinn"]
	# ]
	# axes[-1].legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

	

	# fig.suptitle(
	# 	r"RRMS lesion burden versus non-lesional WM $\mathbf{R_1}/\mathbf{R_2}$",
	# 	fontsize=14,
	# 	fontweight="bold",
	# 	y=0.98,
	# )
	fig.tight_layout(rect=[0, 0.02, 0.88, 0.95])
	return fig


def save_association_summary(df: pd.DataFrame) -> Path | None:
	if df.empty:
		return None

	rows: list[dict[str, object]] = []
	for metric_key, label in [("t1_pinn", "R1"), ("t2_pinn", "R2")]:
		x = df["lesion_volume_ml"].to_numpy(dtype=float)
		y = df[f"{label.lower()}_mean"].to_numpy(dtype=float)
		out = _fit_and_stats(x, y)
		rows.append(
			{
				"metric": label,
				"n": int(np.asarray(out["x"]).size),
				"spearman_rho": out["rho"],
				"spearman_p": out["p"],
				"linear_slope": out["slope"],
				"linear_intercept": out["intercept"],
				"lesion_volume_min": float(np.nanmin(x)),
				"lesion_volume_max": float(np.nanmax(x)),
				"nawm_mean_min": float(np.nanmin(y)),
				"nawm_mean_max": float(np.nanmax(y)),
			}
		)

	out_df = pd.DataFrame(rows)
	out_fp = CACHE_DIR / "rrms_nawm_lesion_volume_association_summary.csv"
	out_df.to_csv(out_fp, index=False)
	print(f"Saved association summary: {out_fp}")
	return out_fp


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def create_figure() -> plt.Figure:
	print("Loading subject mapping...")
	subject_map = get_subject_id_mapping()

	print("Preparing age-matched RRMS table...")
	rrms_df = build_subject_table(subject_map)
	if rrms_df.empty:
		raise RuntimeError("No RRMS subjects with valid lesion-volume and non-lesional WM data were found.")

	print(f"RRMS subjects retained: {len(rrms_df)}")
	print("Saving subject-level values...")
	save_subject_table(rrms_df)
	save_association_summary(rrms_df)

	print("Creating figure...")
	fig = plot_association_panels(rrms_df)

	out_fp = FIG_DIR / f"{EXP_NAME}.svg"
	fig.savefig(out_fp, dpi=300, bbox_inches="tight", facecolor="white")
	print(f"Saved figure: {out_fp}")
	plt.close(fig)
	return fig


if __name__ == "__main__":
	print("=" * 80)
	print(r"RRMS lesion burden versus non-lesional WM $\mathbf{R_1}/\mathbf{R_2}$")
	print("=" * 80)
	create_figure()
	print("Done.")
