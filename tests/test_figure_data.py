"""Consistency checks between the exported tables and the generated figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "experiment_summary.csv"
SWEEP = ROOT / "results" / "lambda_sweep.csv"
PER_HURRICANE = ROOT / "results" / "per_hurricane.csv"
FIGURE_DIR = ROOT / "results"
FIGURE_NAMES = (
    "policy_comparison.png",
    "coverage_by_scenario.png",
    "coverage_by_commodity.png",
    "lambda_pareto.png",
    "response_time_extreme.png",
    "network_map.png",
    "cdf_unmet_demand.png",
)


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    if not SUMMARY.exists():
        pytest.skip("run python -m src.experiments.run_experiments first")
    return pd.read_csv(SUMMARY)


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    if not SWEEP.exists():
        pytest.skip("run python -m src.experiments.run_experiments first")
    return pd.read_csv(SWEEP)


def _row(df: pd.DataFrame, policy: str) -> pd.Series:
    return df.loc[df["policy"] == policy].iloc[0]


def test_policy_row_shapes(summary: pd.DataFrame) -> None:
    assert set(summary["policy"]) == {"risk_neutral", "risk_averse"}
    assert (summary["status"] == "optimal").all()
    assert "solve_time_sec" in summary.columns
    assert (summary["solve_time_sec"] > 0).all()


def test_risk_averse_improves_tail(summary: pd.DataFrame) -> None:
    rn = _row(summary, "risk_neutral")
    ra = _row(summary, "risk_averse")
    assert ra["expected_coverage"] > rn["expected_coverage"]
    assert ra["cvar_shortage"] < rn["cvar_shortage"]
    # risk-averse should always cost at least as much in expectation
    assert ra["expected_cost"] >= rn["expected_cost"] - 1.0


def test_sweep_is_monotone_in_cvar(sweep: pd.DataFrame) -> None:
    """As lambda grows, CVaR of unmet demand must weakly decrease and expected
    cost must weakly increase. This is the defining Pareto property."""
    ordered = sweep.sort_values("lambda").reset_index(drop=True)
    for i in range(1, len(ordered)):
        assert ordered.loc[i, "cvar_shortage"] <= ordered.loc[i - 1, "cvar_shortage"] + 1e-6
        assert ordered.loc[i, "expected_cost"] >= ordered.loc[i - 1, "expected_cost"] - 1e-6


def test_per_hurricane_table_present() -> None:
    if not PER_HURRICANE.exists():
        pytest.skip("run python -m src.experiments.run_experiments first")
    df = pd.read_csv(PER_HURRICANE)
    assert len(df) == 15
    assert {"scenario", "hurricane", "category", "probability", "rn_coverage", "ra_coverage"}.issubset(df.columns)
    assert abs(df["probability"].sum() - 1.0) < 1e-6


def test_figure_files_exist() -> None:
    for name in FIGURE_NAMES:
        assert (FIGURE_DIR / name).exists(), f"missing figure {name}"
