"""Matplotlib figures for the README (static PNGs)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# categorical slots (fixed order) + neutrals
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def calibration(tables: dict[str, pd.DataFrame], path):
    """Reliability diagram for several forecasters."""
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for (name, t), c, mk in zip(tables.items(), (BLUE, ORANGE, AQUA), ("o", "s", "^")):
        t = t[t["n"] >= 20]  # hide near-empty bins
        ax.errorbar(t["p_mean"], t["y_rate"], yerr=[t["y_rate"] - t["lo"], t["hi"] - t["y_rate"]],
                    color=c, marker=mk, ms=6, lw=2, capsize=2, label=name)
    ax.set_xlabel("forecast P(home win)")
    ax.set_ylabel("observed home-win rate (95% CI)")
    ax.set_title("Calibration on 2015-2025 test games", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hfa(by_season: pd.DataFrame, path):
    """Smoothed HFA state with band, plus per-season raw residual estimates."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = by_season["season"]
    ax.fill_between(x, by_season["smooth_lo"], by_season["smooth_hi"], color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, by_season["smooth"], color=BLUE, lw=2, label="model HFA state, smoothed (90% band)")
    ax.plot(x, by_season["filtered_end"], color=VIOLET, lw=2, ls=":", label="real-time filter, end of season")
    ax.errorbar(x, by_season["raw"], yerr=1.96 * by_season["raw_se"], fmt="o", color=ORANGE, ms=6,
                capsize=2, lw=1.2, label="season-only estimate (95% CI)")
    ax.axhline(0, color=MUTED, lw=0.8)
    r20 = by_season.set_index("season").loc[2020, "raw"]
    ax.annotate("2020: no / few fans", xy=(2020, r20), xytext=(2013.8, -1.3),
                arrowprops=dict(arrowstyle="->", color=MUTED), color=INK)
    ax.set_xlabel("season")
    ax.set_ylabel("home-field advantage (points)")
    ax.set_title("Home-field advantage: a 2019-20 dip, then partial recovery", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_xticks(range(2006, 2026, 2))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ratings(traj: pd.DataFrame, teams: list[str], path):
    """Small multiples of smoothed team ratings with 90% bands."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, teams):
        d = traj[traj["team"] == t]
        ax.fill_between(d["t"], d["lo"], d["hi"], color=BLUE, alpha=0.2, lw=0)
        ax.plot(d["t"], d["mean"], color=BLUE, lw=1.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(t, loc="left", color=INK, fontsize=11)
        ax.set_xticks(range(2006, 2027, 4))
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("rating (pts vs avg)")
    fig.suptitle("Smoothed team strength with 90% posterior bands", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def disagreement(ats: pd.DataFrame, breakeven: float, path):
    """ATS win rate of the model's side by size of disagreement with the line."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ats))
    ax.errorbar(x, ats["win_rate"], yerr=[ats["win_rate"] - ats["lo"], ats["hi"] - ats["win_rate"]],
                fmt="o", color=BLUE, ms=8, capsize=3, lw=2)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.axhline(breakeven, color=ORANGE, lw=1.5)
    ax.text(2.5, breakeven + 0.004, "break-even at -110 (52.4%)", color=INK, ha="center", fontsize=9)
    labels = [b.replace("[", "").replace(")", "").replace(".0", "").replace(", ", "-").replace("-inf", "+")
              + f"\nn={n}" for b, n in zip(ats["bucket"], ats["n"])]
    ax.set_xticks(x, labels)
    ax.set_xlabel("|model spread - closing spread| (points)")
    ax.set_ylabel("model side covers (95% CI)")
    ax.set_title("Where the model disagrees with the closing line, 2015-2025", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
