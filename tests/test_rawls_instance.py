"""Structural checks for the Rawls Gulf Coast instance."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.data.rawls_instance import (
    CANDIDATE_HUBS,
    HURRICANES,
    NODES,
    RAW_SCENARIO_PROBS,
    build_rawls_instance,
    load_instance,
    save_instance,
)


@pytest.fixture(scope="module")
def instance() -> dict:
    return build_rawls_instance()


def test_node_and_hub_layout(instance: dict) -> None:
    sets = instance["sets"]
    assert sorted(sets["nodes"]) == list(range(1, 31))
    assert sorted(sets["facilities"]) == sorted(CANDIDATE_HUBS)
    assert len(sets["facilities"]) == 11
    assert set(sets["demand_nodes"]) == set(NODES)
    assert set(sets["transport_modes"]) == {"truck", "air"}
    assert set(sets["commodities"]) == {"water", "food", "medicine"}
    assert sets["periods"] == [1, 2, 3]


def test_arc_graph_is_connected(instance: dict) -> None:
    nodes = list(instance["sets"]["nodes"])
    arcs = [tuple(a) for a in instance["sets"]["arcs"]]
    adj: dict[int, set[int]] = {n: set() for n in nodes}
    for i, j in arcs:
        adj[i].add(j)
        adj[j].add(i)  # arcs are stored directionally but symmetric by construction
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert seen == set(nodes)


def test_probabilities_sum_to_one(instance: dict) -> None:
    total = sum(instance["parameters"]["p_s"].values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    # 15 single-hurricane scenarios (h1..h15)
    assert len(instance["parameters"]["p_s"]) == 15


def test_scenario_hurricane_mapping_covers_all_hurricanes(instance: dict) -> None:
    mapping = instance["meta"]["scenario_hurricane_map"]
    assert set(mapping.values()) == set(HURRICANES.keys())


def test_damage_rules_match_rawls(instance: dict) -> None:
    gamma = instance["parameters"]["gamma_iks"]
    facilities = instance["sets"]["facilities"]
    for s, (h_id, _) in RAW_SCENARIO_PROBS.items():
        hurricane = HURRICANES[h_id]
        landfall = hurricane["landfall"]
        cat = hurricane["category"]
        expected_at_landfall = 0.0 if cat >= 3 else 0.5
        for i in facilities:
            for k in instance["sets"]["commodities"]:
                g = gamma[f"{i},{k},{s}"]
                if i in landfall:
                    assert g == pytest.approx(expected_at_landfall), (
                        f"gamma mismatch at hub {i} scenario {s} (cat {cat})"
                    )
                else:
                    assert g == pytest.approx(1.0)


def test_unusable_links_zero_truck_capacity_only(instance: dict) -> None:
    params = instance["parameters"]
    periods = instance["sets"]["periods"]
    for s, (h_id, _) in RAW_SCENARIO_PROBS.items():
        unusable = HURRICANES[h_id]["unusable_links"]
        for (i, j) in unusable:
            for t in periods:
                # truck capacity must be zero on unusable link in both directions
                assert params["U_ijmts"][f"{i},{j},truck,{t},{s}"] == 0.0
                assert params["U_ijmts"][f"{j},{i},truck,{t},{s}"] == 0.0
                # air stays open
                assert params["U_ijmts"][f"{i},{j},air,{t},{s}"] > 0.0
                assert params["U_ijmts"][f"{j},{i},air,{t},{s}"] > 0.0


def test_period_split_preserves_hurricane_total(instance: dict) -> None:
    demand = instance["parameters"]["d_ikts"]
    for s, (h_id, _) in RAW_SCENARIO_PROBS.items():
        aggregate = {
            "water": HURRICANES[h_id]["water"],
            "food": HURRICANES[h_id]["food"],
            "medicine": HURRICANES[h_id]["medicine"],
        }
        for k, expected in aggregate.items():
            total = sum(
                v for key, v in demand.items()
                if key.endswith(f",{s}") and f",{k}," in f",{key},"
            )
            assert total == pytest.approx(expected), f"demand mismatch {s} {k}"


def test_json_roundtrip(instance: dict, tmp_path: Path) -> None:
    path = tmp_path / "rawls.json"
    save_instance(instance, path)
    loaded = load_instance(path)
    assert loaded["sets"]["nodes"] == instance["sets"]["nodes"]
    assert loaded["parameters"]["p_s"] == instance["parameters"]["p_s"]
