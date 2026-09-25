#!/usr/bin/env python3
"""Create compact diagnostic figures from the released tables.

These figures document the computation and are not copies of manuscript art.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from verify_reported_results import compute_claims


OUTPUT = REPO_ROOT / "outputs" / "figures"
NAVY = "#214761"
RUST = "#9A4F3D"
GOLD = "#C58A2B"
GRAY = "#666666"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11)


def example_figure() -> Path:
    results = pd.read_csv(
        REPO_ROOT / "outputs" / "example" / "example_basin_results.csv",
        dtype={"GAGE_ID": "string"},
    )
    bins = pd.read_csv(
        REPO_ROOT / "outputs" / "example" / "example_conditional_bins.csv",
        dtype={"GAGE_ID": "string"},
    )
    results = results.sort_values("AGGECOREGION")
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 7.0), sharex=False, sharey=False)
    for ax, row in zip(axes.flat, results.itertuples(index=False)):
        subset = bins[bins["GAGE_ID"].astype(str).str.zfill(8).eq(str(row.GAGE_ID).zfill(8))]
        x = subset["q_center"].to_numpy(float)
        y = subset["conditional_variance_rate"].to_numpy(float)
        ax.scatter(x, y, s=17, facecolor="white", edgecolor=NAVY, linewidth=0.8, zorder=3)
        grid = np.geomspace(x.min(), x.max(), 100)
        fitted = row.variance_coefficient * np.power(grid, row.variance_exponent)
        ax.plot(grid, fitted, color=RUST, linewidth=1.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(
            0.04, 0.95, f"{row.AGGECOREGION}\nUSGS {str(row.GAGE_ID).zfill(8)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.2, fontweight="bold",
        )
        ax.text(
            0.96, 0.06, rf"$m={row.variance_exponent:.2f}$",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.2,
        )
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized starting flow, q")
    for ax in axes[:, 0]:
        ax.set_ylabel("Conditional variance rate")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.985, wspace=0.28, hspace=0.28)
    path = OUTPUT / "01_real_gage_estimator_check.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def continental_figure() -> Path:
    derived = REPO_ROOT / "data" / "derived"
    scaling = pd.read_csv(derived / "basin_scaling.csv.gz")
    signed = pd.read_csv(derived / "basin_signed_tails.csv.gz")
    theory = pd.read_csv(derived / "basin_theory_bridge.csv.gz")
    claims = compute_claims()

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    ax = axes[0, 0]
    m = scaling["recession_variance_exponent"].dropna()
    edges = np.linspace(-0.5, 6.5, 57)
    ax.hist(m, bins=edges, density=True, color="#D9E2E7", edgecolor=NAVY, linewidth=0.45)
    ax.axvline(2.0, color=GRAY, linestyle="--", linewidth=1.0)
    ax.axvline(m.median(), color=RUST, linewidth=1.3)
    ax.set(xlabel="Variance exponent, m", ylabel="Basin density", xlim=(-0.5, 6.5))
    panel_label(ax, "a")

    ax = axes[0, 1]
    for column, color, label in (
        ("positive_tail_multiple_of_normal", RUST, "Positive tail"),
        ("negative_tail_multiple_of_normal", NAVY, "Negative tail"),
    ):
        x, y = ecdf(signed[column])
        ax.plot(x, y, color=color, linewidth=1.45, label=label)
    ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set(xlabel="Tail frequency / Gaussian frequency", ylabel="Empirical cumulative probability")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "b")

    ax = axes[1, 0]
    pair = theory.dropna(subset=["m_decline_recomputed", "two_b_recomputed"])
    ax.hexbin(
        pair["two_b_recomputed"], pair["m_decline_recomputed"], gridsize=35,
        mincnt=1, cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "audit_blues", ["#F7FAFC", "#9FB9C8", NAVY]
        ), linewidths=0,
    )
    low = float(min(pair["two_b_recomputed"].quantile(0.005), pair["m_decline_recomputed"].quantile(0.005)))
    high = float(max(pair["two_b_recomputed"].quantile(0.995), pair["m_decline_recomputed"].quantile(0.995)))
    ax.plot([low, high], [low, high], color=GRAY, linestyle="--", linewidth=1.0)
    ax.set(xlabel="Classical prediction, 2b", ylabel="Matched decline exponent, m$_d$", xlim=(low, high), ylim=(low, high))
    panel_label(ax, "c")

    ax = axes[1, 1]
    leads = np.array([1, 2, 3, 5, 7])
    empirical = np.array([claims[f"empirical_shape_crps_gain_{lead}d"] for lead in leads]) * 100.0
    ax.plot(leads, empirical, color=RUST, marker="o", markersize=4.5, linewidth=1.5, label="Empirical shape")
    state_leads = np.array([1, 2, 3])
    state = np.array([claims[f"state_gaussian_crps_gain_{lead}d"] for lead in state_leads]) * 100.0
    ax.plot(state_leads, state, color=NAVY, marker="s", markersize=4.2, linewidth=1.4, label="Flow-state moments")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set(xlabel="Lead time (days)", ylabel="Relative CRPS reduction (%)", xticks=leads)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "d")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.98, wspace=0.28, hspace=0.29)
    path = OUTPUT / "02_continental_result_audit.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def main() -> None:
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [example_figure(), continental_figure()]
    for path in paths:
        print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
