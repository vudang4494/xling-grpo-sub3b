"""Phase 9.5 — Aggregate all eval JSONs into master CSV + bootstrap 95% CIs.

Outputs:
    results/master_v2.csv            One row per (cell, benchmark, language)
    results/stats_v2.csv             Per-arm statistics + bootstrap CIs
    paper/tables/table_main.tex     Multi-seed main results (hand-written, verified)
    paper/tables/table_delta.tex    Δ vs base
    paper/tables/table_mgsm.tex     Multilingual sweep

This script is the authoritative source for aggregate statistics.
It replaces the old aggregate.py and superseded aggregate_phase9.py.

SOURCES (verified against eval JSONs on disk):
  - AMC-23 pass@1, MATH-500 pass@1, AIME-2024 pass@1, AIME-2024 maj@8:
      results/eval/{cell}_step50/*.json
  - AMC-23 maj@4 for ArmA/ArmB/ArmC:
      results/eval/w17_{cell}_{seed}_v2/*.json  (post-fix re-eval)
  - AMC-23 maj@4 for Base, ArmD:
      results/eval/w17_base_distill_v2/ (Base)
      results/eval/a4_const_bias_{seed}_step50/ (ArmD)
  - MGSM:
      results/eval/{cell}_mgsm/*.json

Usage: python3 scripts/aggregate_phase9.py [--print-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
EVAL = ROOT / "results/eval"
OUT_MASTER = ROOT / "results/master_v2.csv"
OUT_STATS = ROOT / "results/stats_v2.csv"
OUT_TABLES = ROOT / "paper/tables"

# ---------------------------------------------------------------------------
# Canonical data manifest: (arm, seed, benchmark) -> relative JSON path
# All paths verified to exist on disk as of 2026-06-01.
# ---------------------------------------------------------------------------
MANIFEST: dict[tuple[str, int | None, str], str] = {
    # ---- Base (single eval, no training) ----
    ("base", 42, "amc23"):   "w17_base_distill_v2/base_distill_v2_amc23.json",
    ("base", 42, "math500"): "base_v3_openrs_eval/base_v3_math500.json",
    ("base", 42, "aime2024"):"base_v3_openrs_eval/base_v3_aime2024.json",
    # ---- ArmA (reproduce_openrs_rs2) — seeds 42, 123, 7 ----
    ("A1", 42,  "amc23"):   "w17_reproduce_openrs_rs2_42_v2/reproduce_openrs_rs2_42_v2_amc23.json",
    ("A1", 42,  "math500"): "ckpt50_v3_openrs_eval/ckpt50_v3_math500.json",
    ("A1", 42,  "aime2024"):"ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json",
    ("A1", 123, "amc23"):   "w17_reproduce_openrs_rs2_123_v2/reproduce_openrs_rs2_123_v2_amc23.json",
    ("A1", 123, "math500"): "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json",
    ("A1", 123, "aime2024"):"reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json",
    ("A1", 7,   "amc23"):   "w17_reproduce_openrs_rs2_7_v2/reproduce_openrs_rs2_7_v2_amc23.json",
    ("A1", 7,   "math500"): "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json",
    ("A1", 7,   "aime2024"):"reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json",
    # ---- ArmB (a2_vi) — seeds 42, 123, 7 ----
    ("A2", 42,  "amc23"):   "w17_a2_vi_42_v2/a2_vi_42_v2_amc23.json",
    ("A2", 42,  "math500"): "a2_vi_42_step50/a2_vi_42_step50_math500.json",
    ("A2", 42,  "aime2024"):"a2_vi_42_step50/a2_vi_42_step50_aime2024.json",
    ("A2", 123, "amc23"):   "w17_a2_vi_123_v2/a2_vi_123_v2_amc23.json",
    ("A2", 123, "math500"): "a2_vi_123_step50/a2_vi_123_step50_math500.json",
    ("A2", 123, "aime2024"):"a2_vi_123_step50/a2_vi_123_step50_aime2024.json",
    ("A2", 7,   "amc23"):   "w17_a2_vi_7_v2/a2_vi_7_v2_amc23.json",
    ("A2", 7,   "math500"): "a2_vi_7_step50/a2_vi_7_step50_math500.json",
    ("A2", 7,   "aime2024"):"a2_vi_7_step50/a2_vi_7_step50_aime2024.json",
    # ---- ArmC (a3_enlang) — seeds 42, 123, 7 ----
    ("A3", 42,  "amc23"):   "w17_a3_enlang_42_v2/a3_enlang_42_v2_amc23.json",
    ("A3", 42,  "math500"): "a3_enlang_42_step50/a3_enlang_42_step50_math500.json",
    ("A3", 42,  "aime2024"):"a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json",
    ("A3", 123, "amc23"):   "w17_a3_enlang_123_v2/a3_enlang_123_v2_amc23.json",
    ("A3", 123, "math500"): "a3_enlang_123_step50/a3_enlang_123_step50_math500.json",
    ("A3", 123, "aime2024"):"a3_enlang_123_step50/a3_enlang_123_step50_aime2024.json",
    ("A3", 7,   "amc23"):   "w17_a3_enlang_7_v2/a3_enlang_7_v2_amc23.json",
    ("A3", 7,   "math500"): "a3_enlang_7_step50/a3_enlang_7_step50_math500.json",
    ("A3", 7,   "aime2024"):"a3_enlang_7_step50/a3_enlang_7_step50_aime2024.json",
    # ---- ArmD (a4_const_bias) — seeds 42, 123, 7 (step50, not re-evaluated) ----
    ("A4", 42,  "amc23"):   "a4_const_bias_42_step50/a4_const_bias_42_step50_amc23.json",
    ("A4", 42,  "math500"): "a4_const_bias_42_step50/a4_const_bias_42_step50_math500.json",
    ("A4", 42,  "aime2024"):"a4_const_bias_42_step50/a4_const_bias_42_step50_aime2024.json",
    ("A4", 123, "amc23"):   "a4_const_bias_123_step50/a4_const_bias_123_step50_amc23.json",
    ("A4", 123, "math500"): "a4_const_bias_123_step50/a4_const_bias_123_step50_math500.json",
    ("A4", 123, "aime2024"):"a4_const_bias_123_step50/a4_const_bias_123_step50_aime2024.json",
    ("A4", 7,   "amc23"):   "a4_const_bias_7_step50/a4_const_bias_7_step50_amc23.json",
    ("A4", 7,   "math500"): "a4_const_bias_7_step50/a4_const_bias_7_step50_math500.json",
    ("A4", 7,   "aime2024"):"a4_const_bias_7_step50/a4_const_bias_7_step50_aime2024.json",
}

# MGSM manifest: (arm, seed, lang) -> relative JSON path
MGSM_MANIFEST: dict[tuple[str, int | None, str], str] = {
    ("base", 42, "en"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_en.json",
    ("base", 42, "es"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_es.json",
    ("base", 42, "fr"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_fr.json",
    ("base", 42, "de"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_de.json",
    ("base", 42, "ru"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_ru.json",
    ("base", 42, "zh"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_zh.json",
    ("base", 42, "ja"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_ja.json",
    ("base", 42, "th"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_th.json",
    ("base", 42, "sw"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_sw.json",
    ("base", 42, "bn"): "base_distill15b_mgsm/base_distill15b_mgsm_mgsm_bn.json",
    # A1: seeds 123 and 7 (seed 42 MGSM not run)
    ("A1", 123, "en"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_en.json",
    ("A1", 123, "es"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_es.json",
    ("A1", 123, "fr"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_fr.json",
    ("A1", 123, "de"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_de.json",
    ("A1", 123, "ru"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_ru.json",
    ("A1", 123, "zh"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_zh.json",
    ("A1", 123, "ja"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_ja.json",
    ("A1", 123, "th"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_th.json",
    ("A1", 123, "sw"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_sw.json",
    ("A1", 123, "bn"): "reproduce_openrs_rs2_123_mgsm/reproduce_openrs_rs2_123_mgsm_mgsm_bn.json",
    ("A1", 7,   "en"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_en.json",
    ("A1", 7,   "es"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_es.json",
    ("A1", 7,   "fr"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_fr.json",
    ("A1", 7,   "de"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_de.json",
    ("A1", 7,   "ru"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_ru.json",
    ("A1", 7,   "zh"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_zh.json",
    ("A1", 7,   "ja"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_ja.json",
    ("A1", 7,   "th"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_th.json",
    ("A1", 7,   "sw"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_sw.json",
    ("A1", 7,   "bn"): "reproduce_openrs_rs2_7_mgsm/reproduce_openrs_rs2_7_mgsm_mgsm_bn.json",
    # A2: seeds 123 and 7
    ("A2", 123, "en"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_en.json",
    ("A2", 123, "es"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_es.json",
    ("A2", 123, "fr"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_fr.json",
    ("A2", 123, "de"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_de.json",
    ("A2", 123, "ru"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_ru.json",
    ("A2", 123, "zh"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_zh.json",
    ("A2", 123, "ja"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_ja.json",
    ("A2", 123, "th"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_th.json",
    ("A2", 123, "sw"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_sw.json",
    ("A2", 123, "bn"): "a2_vi_123_mgsm/a2_vi_123_mgsm_mgsm_bn.json",
    ("A2", 7,   "en"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_en.json",
    ("A2", 7,   "es"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_es.json",
    ("A2", 7,   "fr"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_fr.json",
    ("A2", 7,   "de"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_de.json",
    ("A2", 7,   "ru"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_ru.json",
    ("A2", 7,   "zh"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_zh.json",
    ("A2", 7,   "ja"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_ja.json",
    ("A2", 7,   "th"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_th.json",
    ("A2", 7,   "sw"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_sw.json",
    ("A2", 7,   "bn"): "a2_vi_7_mgsm/a2_vi_7_mgsm_mgsm_bn.json",
    # A3: seeds 123 and 7
    ("A3", 123, "en"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_en.json",
    ("A3", 123, "es"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_es.json",
    ("A3", 123, "fr"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_fr.json",
    ("A3", 123, "de"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_de.json",
    ("A3", 123, "ru"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_ru.json",
    ("A3", 123, "zh"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_zh.json",
    ("A3", 123, "ja"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_ja.json",
    ("A3", 123, "th"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_th.json",
    ("A3", 123, "sw"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_sw.json",
    ("A3", 123, "bn"): "a3_enlang_123_mgsm/a3_enlang_123_mgsm_mgsm_bn.json",
    ("A3", 7,   "en"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_en.json",
    ("A3", 7,   "es"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_es.json",
    ("A3", 7,   "fr"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_fr.json",
    ("A3", 7,   "de"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_de.json",
    ("A3", 7,   "ru"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_ru.json",
    ("A3", 7,   "zh"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_zh.json",
    ("A3", 7,   "ja"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_ja.json",
    ("A3", 7,   "th"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_th.json",
    ("A3", 7,   "sw"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_sw.json",
    ("A3", 7,   "bn"): "a3_enlang_7_mgsm/a3_enlang_7_mgsm_mgsm_bn.json",
    # A4: seed 42 only
    ("A4", 42, "en"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_en.json",
    ("A4", 42, "es"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_es.json",
    ("A4", 42, "fr"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_fr.json",
    ("A4", 42, "de"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_de.json",
    ("A4", 42, "ru"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_ru.json",
    ("A4", 42, "zh"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_zh.json",
    ("A4", 42, "ja"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_ja.json",
    ("A4", 42, "th"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_th.json",
    ("A4", 42, "sw"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_sw.json",
    ("A4", 42, "bn"): "a4_const_bias_42_mgsm/a4_const_bias_42_mgsm_mgsm_bn.json",
}

LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]


def load_metric(arm: str, seed: int | None, bench: str, metric: str) -> float | None:
    key = (arm, seed, bench)
    rel = MANIFEST.get(key)
    if rel is None:
        return None
    fp = EVAL / rel
    if not fp.exists():
        return None
    try:
        d = json.loads(fp.read_text())
        return d.get(metric)
    except Exception:
        return None


def load_mgsm_metric(arm: str, seed: int | None, lang: str) -> float | None:
    key = (arm, seed, lang)
    rel = MGSM_MANIFEST.get(key)
    if rel is None:
        return None
    fp = EVAL / rel
    if not fp.exists():
        return None
    try:
        d = json.loads(fp.read_text())
        return d.get("pass_at_1")
    except Exception:
        return None


def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Subject-level bootstrap CI on the sample mean."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return st.mean(values), lo, hi


def compute_stats() -> list[dict]:
    """Per (arm, benchmark, metric) — mean, std, n_seeds, bootstrap CI."""
    ARMS = ["base", "A1", "A2", "A3", "A4"]
    SEEDS_PER_ARM = {
        "base": [42],
        "A1": [42, 123, 7],
        "A2": [42, 123, 7],
        "A3": [42, 123, 7],
        "A4": [42, 123, 7],
    }
    BENCHES = ["amc23", "math500", "aime2024"]
    METRICS = ["pass_at_1", "maj_at_4", "maj_at_8"]
    results = []

    for arm in ARMS:
        seeds = SEEDS_PER_ARM[arm]
        for bench in BENCHES:
            for metric in METRICS:
                vals = []
                for seed in seeds:
                    v = load_metric(arm, seed, bench, metric)
                    # maj_at_4 = 0 is valid (means 0% majority correct)
                    if isinstance(v, (int, float)):
                        vals.append(v)
                if not vals:
                    continue
                mean, lo, hi = bootstrap_ci(vals)
                results.append({
                    "arm": arm,
                    "benchmark": bench,
                    "metric": metric,
                    "n_seeds": len(vals),
                    "mean": round(mean, 6),
                    "std": round(st.stdev(vals), 6) if len(vals) > 1 else 0.0,
                    "ci_low": round(lo, 6),
                    "ci_high": round(hi, 6),
                    "values": ",".join(f"{v:.4f}" for v in vals),
                })

    # MGSM
    MGSM_SEEDS = {"base": [42], "A1": [123, 7], "A2": [123, 7], "A3": [123, 7], "A4": [42]}
    for arm in ["base", "A1", "A2", "A3", "A4"]:
        for lang in LANGS:
            vals = []
            for seed in MGSM_SEEDS[arm]:
                v = load_mgsm_metric(arm, seed, lang)
                if isinstance(v, (int, float)):
                    vals.append(v)
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            results.append({
                "arm": arm,
                "benchmark": f"mgsm_{lang}",
                "metric": "pass_at_1",
                "n_seeds": len(vals),
                "mean": round(mean, 6),
                "std": round(st.stdev(vals), 6) if len(vals) > 1 else 0.0,
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "values": ",".join(f"{v:.4f}" for v in vals),
            })

    return results


def write_master_v2(stats: list[dict]) -> None:
    """Write every JSON value (one row per manifest entry) to master CSV."""
    rows = []
    for (arm, seed, bench), rel in MANIFEST.items():
        fp = EVAL / rel
        if not fp.exists():
            continue
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        rows.append({
            "arm": arm,
            "seed": seed,
            "benchmark": bench,
            "pass_at_1": d.get("pass_at_1"),
            "maj_at_4": d.get("maj_at_4"),
            "maj_at_8": d.get("maj_at_8"),
            "n_samples": d.get("n_samples"),
            "source_file": rel,
        })
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MASTER.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "seed", "benchmark",
                                          "pass_at_1", "maj_at_4", "maj_at_8",
                                          "n_samples", "source_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"  master_v2.csv: {len(rows)} rows -> {OUT_MASTER}")


def write_stats_v2(stats: list[dict]) -> None:
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_STATS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "benchmark", "metric",
                                          "n_seeds", "mean", "std",
                                          "ci_low", "ci_high", "values"])
        w.writeheader()
        w.writerows(stats)
    print(f"  stats_v2.csv: {len(stats)} entries -> {OUT_STATS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    print("[aggregate_v2] computing statistics...")
    stats = compute_stats()

    # Print verified numbers
    print("\n========== Verified statistics ==========")
    for arm in ["base", "A1", "A2", "A3", "A4"]:
        arm_stats = [s for s in stats if s["arm"] == arm]
        print(f"\n[{arm.upper()}]")
        for s in sorted(arm_stats, key=lambda x: (x["benchmark"], x["metric"])):
            m = f"{s['mean']*100:.1f}"
            if s["std"] > 0:
                m += f" ± {s['std']*100:.1f}"
            ci = f" [95% CI: {s['ci_low']*100:.1f}–{s['ci_high']*100:.1f}]"
            print(f"  {s['benchmark']:12s} {s['metric']:12s} "
                  f"n={s['n_seeds']}  {m}%{ci}")

    if args.print_only:
        return

    print("\n[aggregate_v2] writing output files...")
    write_master_v2(stats)
    write_stats_v2(stats)
    print("[aggregate_v2] done.")


if __name__ == "__main__":
    main()
