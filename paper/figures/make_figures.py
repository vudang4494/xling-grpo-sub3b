"""Multi-seed figures for the paper.

Reads all eval JSONs via the canonical MANIFEST in aggregate_phase9.py
and emits 6 publication-grade PDFs (300 DPI, embedded fonts) under
paper/figures/.

Data sources:
  - AMC-23 pass@1, MATH-500, AIME-2024 (all arms, all seeds):
      results/eval/{cell}_step50/*.json
  - AMC-23 maj@4 for ArmA/ArmB/ArmC:
      results/eval/w17_{cell}_{seed}_v2/*.json   (W17 post-bugfix re-eval)
  - AMC-23 maj@4 for Base, ArmD:
      results/eval/w17_base_distill_v2/ (Base)
      results/eval/a4_const_bias_{seed}_step50/ (ArmD)
  - MGSM:
      results/eval/{cell}_mgsm/*.json

NOTE: Figure 5 (training curves) uses 2 seeds (123, 7) per arm.
This is because seed-42 checkpoint logs were not saved for A1/A2/A3 arms.
See the figure caption for this limitation.

Usage: python paper/figures/make_figures.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
EVAL_DIR = ROOT / "results/eval"
GRPO_DIR = ROOT / "results/grpo"
OUT_DIR = ROOT / "paper/figures"

# ---------------------------------------------------------------------------
# Explicit data manifest — mirrors aggregate_phase9.py MANIFEST exactly.
# Every (arm, seed, bench) -> (metric, value) is read from these files.
# This is the ONLY place figure data is defined.
# ---------------------------------------------------------------------------
CELLS: dict[str, dict[str, Path | None]] = {
    # base: single eval
    "Base": {"amc23": EVAL_DIR / "w17_base_distill_v2/base_distill_v2_amc23.json",
             "math500": EVAL_DIR / "base_v3_openrs_eval/base_v3_math500.json",
             "aime2024": EVAL_DIR / "base_v3_openrs_eval/base_v3_aime2024.json"},
    # ArmA (reproduce_openrs_rs2)
    "A1 s42": {"amc23": EVAL_DIR / "w17_reproduce_openrs_rs2_42_v2/reproduce_openrs_rs2_42_v2_amc23.json",
                "math500": EVAL_DIR / "ckpt50_v3_openrs_eval/ckpt50_v3_math500.json",
                "aime2024": EVAL_DIR / "ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json"},
    "A1 s123": {"amc23": EVAL_DIR / "w17_reproduce_openrs_rs2_123_v2/reproduce_openrs_rs2_123_v2_amc23.json",
                 "math500": EVAL_DIR / "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json",
                 "aime2024": EVAL_DIR / "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json"},
    "A1 s7":   {"amc23": EVAL_DIR / "w17_reproduce_openrs_rs2_7_v2/reproduce_openrs_rs2_7_v2_amc23.json",
                 "math500": EVAL_DIR / "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json",
                 "aime2024": EVAL_DIR / "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json"},
    # ArmB (a2_vi)
    "A2 s42": {"amc23": EVAL_DIR / "w17_a2_vi_42_v2/a2_vi_42_v2_amc23.json",
                "math500": EVAL_DIR / "a2_vi_42_step50/a2_vi_42_step50_math500.json",
                "aime2024": EVAL_DIR / "a2_vi_42_step50/a2_vi_42_step50_aime2024.json"},
    "A2 s123": {"amc23": EVAL_DIR / "w17_a2_vi_123_v2/a2_vi_123_v2_amc23.json",
                 "math500": EVAL_DIR / "a2_vi_123_step50/a2_vi_123_step50_math500.json",
                 "aime2024": EVAL_DIR / "a2_vi_123_step50/a2_vi_123_step50_aime2024.json"},
    "A2 s7":   {"amc23": EVAL_DIR / "w17_a2_vi_7_v2/a2_vi_7_v2_amc23.json",
                 "math500": EVAL_DIR / "a2_vi_7_step50/a2_vi_7_step50_math500.json",
                 "aime2024": EVAL_DIR / "a2_vi_7_step50/a2_vi_7_step50_aime2024.json"},
    # ArmC (a3_enlang)
    "A3 s42": {"amc23": EVAL_DIR / "w17_a3_enlang_42_v2/a3_enlang_42_v2_amc23.json",
                "math500": EVAL_DIR / "a3_enlang_42_step50/a3_enlang_42_step50_math500.json",
                "aime2024": EVAL_DIR / "a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json"},
    "A3 s123": {"amc23": EVAL_DIR / "w17_a3_enlang_123_v2/a3_enlang_123_v2_amc23.json",
                 "math500": EVAL_DIR / "a3_enlang_123_step50/a3_enlang_123_step50_math500.json",
                 "aime2024": EVAL_DIR / "a3_enlang_123_step50/a3_enlang_123_step50_aime2024.json"},
    "A3 s7":   {"amc23": EVAL_DIR / "w17_a3_enlang_7_v2/a3_enlang_7_v2_amc23.json",
                 "math500": EVAL_DIR / "a3_enlang_7_step50/a3_enlang_7_step50_math500.json",
                 "aime2024": EVAL_DIR / "a3_enlang_7_step50/a3_enlang_7_step50_aime2024.json"},
    # ArmD (a4_const_bias) — all 3 seeds, step50 source throughout
    "A4 s42": {"amc23": EVAL_DIR / "a4_const_bias_42_step50/a4_const_bias_42_step50_amc23.json",
                "math500": EVAL_DIR / "a4_const_bias_42_step50/a4_const_bias_42_step50_math500.json",
                "aime2024": EVAL_DIR / "a4_const_bias_42_step50/a4_const_bias_42_step50_aime2024.json"},
    "A4 s123": {"amc23": EVAL_DIR / "a4_const_bias_123_step50/a4_const_bias_123_step50_amc23.json",
                 "math500": EVAL_DIR / "a4_const_bias_123_step50/a4_const_bias_123_step50_math500.json",
                 "aime2024": EVAL_DIR / "a4_const_bias_123_step50/a4_const_bias_123_step50_aime2024.json"},
    "A4 s7":   {"amc23": EVAL_DIR / "a4_const_bias_7_step50/a4_const_bias_7_step50_amc23.json",
                 "math500": EVAL_DIR / "a4_const_bias_7_step50/a4_const_bias_7_step50_math500.json",
                 "aime2024": EVAL_DIR / "a4_const_bias_7_step50/a4_const_bias_7_step50_aime2024.json"},
}

ARMS = ["A1", "A2", "A3", "A4"]
ARM_COLORS = {"A1": "#d62728", "A2": "#2ca02c", "A3": "#1f77b4", "A4": "#9467bd", "Base": "#7f7f7f"}
METRIC_LABELS = {
    "AMC": "AMC-23 pass@1",
    "AMC@4": "AMC-23 maj@4",
    "MATH": "MATH-500 pass@1",
    "AIME": "AIME-2024 pass@1",
    "AIME@8": "AIME-2024 maj@8",
}
METRICS = list(METRIC_LABELS.keys())

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
})


def _read_json(fp: Path) -> dict | None:
    try:
        return json.loads(fp.read_text()) if fp.exists() else None
    except Exception:
        return None


def load_data() -> dict[str, dict[str, float | None]]:
    """Return {cell_label: {metric: value (0-1 or None)}}."""
    data: dict[str, dict[str, float | None]] = {}
    for label, benches in CELLS.items():
        row: dict[str, float | None] = {m: None for m in METRICS}
        for bench_key, fp in benches.items():
            d = _read_json(fp)
            if not d:
                continue
            p1 = d.get("pass_at_1")
            m4 = d.get("maj_at_4")
            m8 = d.get("maj_at_8")
            if bench_key == "amc23":
                if isinstance(p1, (int, float)):
                    row["AMC"] = p1
                if isinstance(m4, (int, float)):
                    row["AMC@4"] = m4
            elif bench_key == "math500":
                if isinstance(p1, (int, float)):
                    row["MATH"] = p1
            elif bench_key == "aime2024":
                if isinstance(p1, (int, float)):
                    row["AIME"] = p1
                if isinstance(m8, (int, float)):
                    row["AIME@8"] = m8
        data[label] = row
    return data


def arm_vals(data: dict, arm: str, metric: str) -> tuple[list[float], float | None, float | None]:
    """Return (per-cell values as %, mean%, std%)."""
    vals = []
    for label, row in data.items():
        if label.startswith(arm) and isinstance(row.get(metric), (int, float)):
            vals.append(row[metric] * 100)
    if len(vals) < 2:
        return vals, (vals[0] if vals else None), None
    return vals, st.mean(vals), st.stdev(vals)


def fig1_arm_means(data: dict) -> None:
    """Bar chart: mean ± σ across all seeds per arm, 5 metrics."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n_metrics = len(METRICS)
    bar_width = 0.2
    x = np.arange(n_metrics)

    for i, arm in enumerate(ARMS):
        means, errs = [], []
        for m in METRICS:
            _, mean, std = arm_vals(data, arm, m)
            means.append(mean if mean is not None else 0)
            errs.append(std if std is not None else 0)
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=errs, capsize=3,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5,
               label=f"{arm} (3 seeds)")

    base_row = data.get("Base", {})
    for j, m in enumerate(METRICS):
        v = base_row.get(m)
        if isinstance(v, (int, float)):
            ax.hlines(v * 100, x[j] - 0.45, x[j] + 0.45,
                      colors="black", linestyles="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Mean ± σ across seeds per arm. Black dashed: untrained base.\n"
                 "AMC-23 maj@4 uses W17 post-bugfix re-evaluations.")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_arm_means_with_ci.pdf", bbox_inches="tight")
    plt.close()
    print("  fig1_arm_means_with_ci.pdf")


def fig2_seed_scatter(data: dict) -> None:
    """Scatter: every per-seed dot, mean bar, ±σ error bar."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 3.5))
    base_row = data.get("Base", {})
    for ax, m in zip(axes, METRICS):
        for i, arm in enumerate(ARMS):
            vals, mean, std = arm_vals(data, arm, m)
            x_pos = i + 1
            ax.scatter([x_pos] * len(vals), vals, s=80,
                       color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5, zorder=3)
            if mean is not None:
                ax.hlines(mean, x_pos - 0.25, x_pos + 0.25,
                          colors=ARM_COLORS[arm], linewidth=2.5)
            if std is not None:
                ax.vlines(x_pos, mean - std, mean + std,
                          colors=ARM_COLORS[arm], linewidth=1.5, alpha=0.6)
        bv = base_row.get(m)
        if isinstance(bv, (int, float)):
            ax.axhline(bv * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                       label="Base")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(ARMS)
        ax.set_title(METRIC_LABELS[m], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0.5, 4.7)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle("Per-seed values: each dot = one of three random seeds.\n"
                 "Bars = mean. Vertical lines = ±1 σ. Dashed = untrained base.\n"
                 "NOTE: A4 not shown in figure legend (added in post).",
                 fontsize=10, y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_seed_scatter.pdf", bbox_inches="tight")
    plt.close()
    print("  fig2_seed_scatter.pdf")


def fig3_variance_ratios(data: dict) -> None:
    """σ per arm per metric — the variance story."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sigmas = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for m in METRICS:
            _, _, s = arm_vals(data, arm, m)
            sigmas[arm].append(s if s is not None else 0)

    bar_width = 0.22
    x = np.arange(len(METRICS))
    for i, arm in enumerate(ARMS):
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, sigmas[arm], bar_width,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5, label=arm)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("σ across seeds (pp)")
    ax.set_title("Seed variance per arm: lower = more reproducible.\n"
                 "A1 (vanilla EN GRPO) consistently shows highest variance.")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_variance_ratios.pdf", bbox_inches="tight")
    plt.close()
    print("  fig3_variance_ratios.pdf")


def fig4_effect_vs_base(data: dict) -> None:
    """Δ from base (mean ± σ) per arm per metric."""
    base_row = data.get("Base", {})
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bar_width = 0.22
    x = np.arange(len(METRICS))

    for i, arm in enumerate(ARMS):
        deltas, errs = [], []
        for m in METRICS:
            _, mean, std = arm_vals(data, arm, m)
            base = base_row.get(m)
            if isinstance(base, (int, float)) and mean is not None:
                deltas.append(mean - base * 100)
                errs.append(std if std is not None else 0)
            else:
                deltas.append(0)
                errs.append(0)
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, deltas, bar_width, yerr=errs, capsize=3,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5, label=arm)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Δ from base (pp)")
    ax.set_title("Effect size relative to untrained base, mean ± σ across seeds.\n"
                 "Positive = improvement; negative = degradation.")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_effect_vs_base.pdf", bbox_inches="tight")
    plt.close()
    print("  fig4_effect_vs_base.pdf")


def load_trainer_history(cell_dir: str) -> list[dict] | None:
    """Read trainer_state.json for a cell. Prefers checkpoint-100, falls back."""
    for ckpt in ("checkpoint-100", "checkpoint-50"):
        p = GRPO_DIR / cell_dir / ckpt / "trainer_state.json"
        if p.exists():
            try:
                return json.loads(p.read_text()).get("log_history", [])
            except Exception:
                pass
    return None


def fig5_training_curves(data: dict) -> None:
    """Training dynamics: mean ± σ across 2 seeds (123, 7) per arm, 100 steps.

    LIMITATION: Only seeds 123 and 7 are available for A1/A2/A3 because
    seed-42 checkpoint logs were not saved for those arms.
    Arm D (a4_const_bias) training logs are not included.
    """
    arm_to_dirs = {
        "A1": ["reproduce_openrs_rs2_123", "reproduce_openrs_rs2_7"],
        "A2": ["a2_vi_123", "a2_vi_7"],
        "A3": ["a3_enlang_123", "a3_enlang_7"],
    }
    panels = ["reward", "kl", "completion_length"]
    panel_titles = {
        "reward": "Total reward",
        "kl": "KL divergence",
        "completion_length": "Completion length (tokens)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for ax, panel in zip(axes, panels):
        for arm, dirs in arm_to_dirs.items():
            histories = []
            for d in dirs:
                h = load_trainer_history(d)
                if h:
                    histories.append(h)
            if not histories:
                continue
            steps = sorted(set(e["step"] for h in histories for e in h))
            ys = []
            for s in steps:
                vs = [e[panel] for h in histories for e in h
                      if e.get("step") == s and panel in e]
                if vs:
                    ys.append((s, vs))
            if not ys:
                continue
            xs = [y[0] for y in ys]
            means = [st.mean(y[1]) for y in ys]
            stds = [st.stdev(y[1]) if len(y[1]) > 1 else 0 for y in ys]
            ax.plot(xs, means, color=ARM_COLORS[arm], label=arm, linewidth=1.7)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=ARM_COLORS[arm], alpha=0.18)
        ax.set_title(panel_titles[panel])
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")
    fig.suptitle("Training dynamics (mean ± σ across 2 seeds: 123, 7).\n"
                 "LIMITATION: seed-42 logs not saved for A1/A2/A3; Arm D excluded.",
                 fontsize=10, y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_training_curves_multiseed.pdf", bbox_inches="tight")
    plt.close()
    print("  fig5_training_curves_multiseed.pdf")


def fig6_aime_focus(data: dict) -> None:
    """AIME-2024 per-seed breakdown: hardest benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    base_row = data.get("Base", {})

    for ax, m in zip(axes, ["AIME", "AIME@8"]):
        bv = base_row.get(m)
        if isinstance(bv, (int, float)):
            ax.axhline(bv * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"Base = {bv * 100:.1f}%")
        for i, arm in enumerate(ARMS):
            vals, mean, std = arm_vals(data, arm, m)
            x_pos = i + 1
            ax.scatter([x_pos] * len(vals), vals, s=110,
                       color=ARM_COLORS[arm], edgecolor="black", linewidth=0.6, zorder=3, alpha=0.85)
            if mean is not None:
                ax.hlines(mean, x_pos - 0.3, x_pos + 0.3, colors=ARM_COLORS[arm], linewidth=3)
                ax.text(x_pos + 0.32, mean, f"{mean:.1f}", color=ARM_COLORS[arm],
                        fontsize=9, va="center", fontweight="bold")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(ARMS)
        ax.set_title(METRIC_LABELS[m])
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0.5, 4.7)
        ax.legend(loc="upper left")
    fig.suptitle("AIME-2024 (hardest benchmark): per-seed values.\n"
                 "Arm C achieves highest mean and lowest variance on maj@8.",
                 fontsize=10, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_aime_focus.pdf", bbox_inches="tight")
    plt.close()
    print("  fig6_aime_focus.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figures] reading from {EVAL_DIR}")
    data = load_data()
    print(f"[figures] loaded {len(data)} cells")
    for cell, row in data.items():
        non_null = sum(1 for v in row.values() if v is not None)
        print(f"  {cell}: {non_null}/{len(METRICS)} metrics")
    print("[figures] generating PDFs:")
    fig1_arm_means(data)
    fig2_seed_scatter(data)
    fig3_variance_ratios(data)
    fig4_effect_vs_base(data)
    fig5_training_curves(data)
    fig6_aime_focus(data)
    print(f"[figures] done. wrote 6 PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
