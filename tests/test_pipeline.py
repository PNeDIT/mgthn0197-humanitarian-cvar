"""End-to-end pipeline tests for the Gurobi model on the Rawls instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.rawls_instance import build_rawls_instance, load_instance, save_instance
from src.models.cvar_model import HumanitarianCVaRModel, solve_instance

pytest.importorskip("gurobipy")


@pytest.fixture(scope="module")
def instance() -> dict:
    return build_rawls_instance()


def test_instance_structure(instance: dict) -> None:
    sets = instance["sets"]
    params = instance["parameters"]
    assert len(sets["nodes"]) == 30
    assert len(sets["facilities"]) == 11
    assert len(sets["scenarios"]) == 15
    assert set(sets["transport_modes"]) == {"truck", "air"}
    assert abs(sum(params["p_s"].values()) - 1.0) < 1e-9


def test_instance_json_roundtrip(tmp_path: Path, instance: dict) -> None:
    path = tmp_path / "inst.json"
    save_instance(instance, path)
    loaded = load_instance(path)
    assert loaded["sets"]["transport_modes"] == instance["sets"]["transport_modes"]
    assert loaded["parameters"]["p_s"] == instance["parameters"]["p_s"]


def test_risk_neutral_solves(instance: dict) -> None:
    result = solve_instance(instance, lambda_risk=0.0)
    assert result.status == "optimal"
    assert result.solve_time_sec > 0
    assert 0 <= result.expected_coverage <= 1
    # every scenario must have coverage in [0, 1] and non-negative unmet demand
    for s in instance["sets"]["scenarios"]:
        assert 0 <= result.coverage_by_scenario[s] <= 1 + 1e-9
        assert result.unmet_by_scenario[s] >= -1e-6


def test_risk_averse_solves(instance: dict) -> None:
    result = solve_instance(instance, lambda_risk=18.0, alpha=0.95)
    assert result.status == "optimal"
    assert result.cvar_value >= -1e-6


def test_policies_differ_on_tail_metrics(instance: dict) -> None:
    rn = solve_instance(instance, lambda_risk=0.0)
    ra = solve_instance(instance, lambda_risk=18.0, alpha=0.95)
    # the risk-averse policy must weakly reduce CVaR of shortages and improve
    # coverage on the scenario the risk-neutral policy handles worst
    assert ra.cvar_value <= rn.cvar_value + 1e-6
    worst = min(rn.coverage_by_scenario.items(), key=lambda kv: kv[1])[0]
    assert ra.coverage_by_scenario[worst] > rn.coverage_by_scenario[worst] + 1e-6
    assert ra.expected_coverage >= rn.expected_coverage - 1e-6
    # the risk-averse policy should not open fewer hubs than the risk-neutral one
    assert len(ra.open_facilities) >= len(rn.open_facilities)


def test_per_commodity_metrics_present(instance: dict) -> None:
    result = solve_instance(instance, lambda_risk=0.0)
    for s in instance["sets"]["scenarios"]:
        for k in instance["sets"]["commodities"]:
            cov = result.coverage_by_commodity[s][k]
            assert 0 <= cov <= 1 + 1e-9


def test_inventory_balance_feasible(instance: dict) -> None:
    model = HumanitarianCVaRModel(instance, lambda_risk=0.0)
    model.build()
    model.solve()
    m = model.model
    assert m is not None
    for constr in m.getConstrs():
        if constr.ConstrName.startswith("balance_"):
            assert abs(constr.Slack) < 1e-4
