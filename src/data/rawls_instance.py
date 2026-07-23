"""Rawls & Turnquist (2010) Gulf Coast test instance.

This module encodes the 30-node Gulf Coast network and the 15 single-hurricane
scenarios from Rawls and Turnquist (2010), Tables 3 and 4. Values that are not
explicitly tabulated in the source paper (e.g. exact node coordinates, arc
adjacency, per-arc truck capacities, air-mode data, and the split of each
hurricane's demand across operational periods) are reconstructed here and every
reconstruction is documented in-line so the case study can be defended.

Key design points
-----------------
* All 30 Rawls nodes are potential demand locations. 11 of them are candidate
  hubs; the selection includes both coastal (vulnerable) and inland (safer)
  locations so the model has a meaningful open/close trade-off.
* Damage rule follows Rawls: hurricanes of category 3 or higher destroy all
  supplies at their landfall node(s) (gamma = 0); category 1 or 2 damage 50%
  of the supplies at the landfall node(s) (gamma = 0.5). Non-landfall nodes
  are unaffected (gamma = 1).
* "Links unusable" from Rawls Table 3 zero out the truck capacity on the
  corresponding arcs in both directions. Air transport is not affected by
  unusable roads and remains available.
* The 15 single-hurricane scenarios (Rawls Table 4, top block) sum to 0.75 in
  the source paper because the remaining 0.25 mass covers combined-hurricane
  scenarios that we do not reproduce. Probabilities are renormalized by /0.75
  so that they sum to 1 on our restricted scenario set.
* Yi and Ozdamar (2007) motivate the multi-period extension. Each hurricane's
  aggregate demand is distributed across three operational periods with weights
  (0.50, 0.30, 0.20) to represent the immediate, short-term and sustained
  phases of the response.

The module exposes :func:`build_rawls_instance` which returns a dictionary
consumed directly by the Gurobi model in :mod:`src.models.cvar_model`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SEED = 2010
TRANSPORT_MODES: tuple[str, str] = ("truck", "air")
FACILITY_SIZES: tuple[str, str, str] = ("small", "medium", "large")
COMMODITIES: tuple[str, str, str] = ("water", "food", "medicine")
PERIODS: tuple[int, int, int] = (1, 2, 3)
PERIOD_WEIGHTS: dict[int, float] = {1: 0.50, 2: 0.30, 3: 0.20}
ARC_DISTANCE_THRESHOLD: float = 6.5


# -- Node layout ------------------------------------------------------------

# Approximate Gulf Coast layout. The x-axis roughly spans the coastline from
# south Texas (small x) to north Florida (large x); y grows from the coast
# inland. Landfall nodes from Rawls Table 3 are placed on the coastal band so
# they carry the physical semantics of a coastal impact zone. The exact
# coordinates are a reconstruction; only their relative geometry matters for
# the model (arcs are built from distances).
NODE_COORDINATES: dict[int, tuple[float, float]] = {
    # coastal band (y <= 1.5) -- includes all Rawls landfall nodes
    3: (5.5, 1.0),
    4: (6.5, 1.6),
    5: (7.5, 1.0),
    11: (16.0, 1.0),
    13: (19.0, 1.0),
    14: (20.5, 1.4),
    15: (22.0, 1.0),
    21: (14.0, 1.4),
    22: (17.0, 1.2),
    29: (16.5, 1.6),
    30: (20.0, 1.6),
    # mid band (1.5 < y <= 3.5)
    1: (1.0, 2.5),
    6: (9.0, 2.4),
    8: (12.0, 2.6),
    10: (14.5, 2.6),
    12: (17.5, 2.6),
    17: (4.5, 3.2),
    20: (11.5, 3.4),
    23: (19.5, 3.2),
    26: (6.0, 3.4),
    28: (12.5, 3.4),
    # inland band (y > 3.5) -- safer for pre-positioning
    2: (3.0, 4.5),
    7: (10.5, 4.7),
    9: (13.5, 5.0),
    16: (2.0, 5.2),
    18: (7.0, 4.8),
    19: (9.5, 5.0),
    24: (21.5, 4.8),
    25: (3.5, 5.6),
    27: (9.0, 5.6),
}

NODES: list[int] = sorted(NODE_COORDINATES.keys())
assert NODES == list(range(1, 31)), "Rawls layout must cover nodes 1..30"

# 11 candidate hubs: mix of safe (inland) and vulnerable (near coast) locations
# so the model must trade robustness against proximity to demand.
CANDIDATE_HUBS: list[int] = sorted([2, 6, 8, 11, 16, 18, 20, 22, 25, 27, 28])


# -- Rawls hurricane characteristics (Table 3) -----------------------------

# Each entry: (category, landfall_nodes, unusable_links,
#              water_demand_1000gal, food_demand_1000units, medicine_units).
# Empty landfall_nodes means the hurricane causes distributed demand but no
# specific coastal strike (Rawls scenarios 10, 12, 13).
HURRICANES: dict[int, dict[str, Any]] = {
    1:  {"category": 3, "landfall": [3],       "unusable_links": [(4, 5)],
         "water": 350.0,    "food": 525.0,   "medicine": 500.0},
    2:  {"category": 5, "landfall": [14],      "unusable_links": [(12, 14), (14, 15), (15, 24)],
         "water": 560.0,    "food": 927.0,   "medicine": 883.0},
    3:  {"category": 2, "landfall": [22],      "unusable_links": [],
         "water": 861.0,    "food": 181.0,   "medicine": 402.0},
    4:  {"category": 2, "landfall": [22],      "unusable_links": [(17, 20)],
         "water": 9000.0,   "food": 1692.0,  "medicine": 3760.0},
    5:  {"category": 4, "landfall": [11, 29],  "unusable_links": [],
         "water": 7500.0,   "food": 1771.0,  "medicine": 1687.0},
    6:  {"category": 3, "landfall": [15],      "unusable_links": [],
         "water": 1000.0,   "food": 1838.0,  "medicine": 1751.0},
    7:  {"category": 2, "landfall": [21],      "unusable_links": [(21, 22)],
         "water": 600.0,    "food": 324.0,   "medicine": 720.0},
    8:  {"category": 1, "landfall": [11],      "unusable_links": [(8, 12)],
         "water": 1500.0,   "food": 162.0,   "medicine": 360.0},
    9:  {"category": 5, "landfall": [13, 29],  "unusable_links": [(12, 13)],
         "water": 1040.0,   "food": 13300.0, "medicine": 95000.0},
    10: {"category": 2, "landfall": [],        "unusable_links": [],
         "water": 2250.0,   "food": 1125.0,  "medicine": 18750.0},
    11: {"category": 3, "landfall": [21],      "unusable_links": [(21, 22), (15, 24)],
         "water": 5000.0,   "food": 1750.0,  "medicine": 12500.0},
    12: {"category": 3, "landfall": [],        "unusable_links": [],
         "water": 18000.0,  "food": 630.0,   "medicine": 4500.0},
    13: {"category": 3, "landfall": [],        "unusable_links": [],
         "water": 2818.0,   "food": 80.0,    "medicine": 571.0},
    14: {"category": 4, "landfall": [14, 30],  "unusable_links": [],
         "water": 2239.0,   "food": 1477.0,  "medicine": 10551.0},
    15: {"category": 4, "landfall": [22],      "unusable_links": [],
         "water": 4400.0,   "food": 3921.0,  "medicine": 28007.0},
}


# -- Rawls scenario probabilities (Table 4, single-hurricane block) --------

# Scenario -> (hurricane_id, raw_probability). Raw probabilities sum to 0.75
# because the remaining 0.25 mass covers combined-hurricane events that are
# out of scope here. Values are taken verbatim from Rawls Table 4.
RAW_SCENARIO_PROBS: dict[str, tuple[int, float]] = {
    "h1":  (1,  0.02308),
    "h5":  (5,  0.05000),
    "h10": (10, 0.16167),
    "h3":  (3,  0.05363),
    "h2":  (2,  0.00925),
    "h12": (12, 0.03083),
    "h13": (13, 0.13380),
    "h4":  (4,  0.05363),
    "h11": (11, 0.02295),
    "h14": (14, 0.02295),
    "h15": (15, 0.02295),
    "h7":  (7,  0.05363),
    "h9":  (9,  0.05000),
    "h8":  (8,  0.03080),
    "h6":  (6,  0.03083),
}


# -- Commodity economics (reconstructed) -----------------------------------

# Storage space (b_k) and acquisition cost (q_k) are chosen to reflect the
# very different volumes and unit values of the three commodities: water is
# bulky and cheap, food is moderate, medicine is compact and expensive.
COMMODITY_ECONOMICS: dict[str, dict[str, float]] = {
    "water":    {"b": 1.00, "q": 4.0,  "e": 1.00, "h": 0.05, "rho": 20.0},
    "food":     {"b": 0.80, "q": 12.0, "e": 0.80, "h": 0.10, "rho": 60.0},
    "medicine": {"b": 0.10, "q": 25.0, "e": 0.10, "h": 0.20, "rho": 150.0},
}

# Facility size cost/capacity ladder, scaled to the Rawls demand magnitudes.
SIZE_CAPACITY: dict[str, float] = {"small": 6_000.0, "medium": 15_000.0, "large": 40_000.0}
SIZE_FIXED_COST: dict[str, float] = {"small": 8_000.0, "medium": 18_000.0, "large": 45_000.0}


# -- Helpers ----------------------------------------------------------------


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_arcs(nodes: Iterable[int], threshold: float) -> list[tuple[int, int]]:
    """Return a symmetric arc list.

    Includes every ordered pair (i, j) whose Euclidean distance is at most
    ``threshold`` plus every arc that Rawls lists as unusable in at least one
    hurricane (both directions). Forcing the Rawls-mentioned arcs into the
    graph guarantees that "link (i,j) becomes unusable in hurricane h" has a
    well-defined effect on the model regardless of the distance heuristic.
    The graph must remain connected; ``_is_connected`` verifies this after
    construction.
    """
    node_list = sorted(nodes)
    arc_set: set[tuple[int, int]] = set()
    for i in node_list:
        for j in node_list:
            if i == j:
                continue
            if _euclid(NODE_COORDINATES[i], NODE_COORDINATES[j]) <= threshold:
                arc_set.add((i, j))
    for h in HURRICANES.values():
        for (i, j) in h["unusable_links"]:
            if i in NODE_COORDINATES and j in NODE_COORDINATES:
                arc_set.add((i, j))
                arc_set.add((j, i))
    return sorted(arc_set)


def _is_connected(nodes: list[int], arcs: list[tuple[int, int]]) -> bool:
    adj: dict[int, set[int]] = {n: set() for n in nodes}
    for i, j in arcs:
        adj[i].add(j)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)


def _normalize_probs(raw: dict[str, tuple[int, float]]) -> dict[str, float]:
    total = sum(p for _, p in raw.values())
    if total <= 0:
        raise ValueError("Scenario probabilities must have positive mass")
    return {s: p / total for s, (_, p) in raw.items()}


def _demand_split(total: float) -> dict[int, float]:
    """Distribute a hurricane's aggregate demand across the operational
    periods using the (0.50, 0.30, 0.20) split motivated by Yi and Ozdamar.
    """
    return {t: total * PERIOD_WEIGHTS[t] for t in PERIODS}


def _damage_gamma(category: int, landfall: list[int], node: int) -> float:
    if node not in landfall:
        return 1.0
    if category >= 3:
        return 0.0
    return 0.5


def _demand_targets(landfall: list[int], all_demand_nodes: list[int]) -> list[int]:
    """Where a hurricane's demand materializes.

    * If the hurricane has one or more landfall nodes we place the demand at
      those nodes (evenly split when there are several); this matches Rawls'
      interpretation of a hurricane striking specific counties.
    * If Rawls' scenario has no landfall (hurricanes 10, 12, 13) demand is
      distributed evenly across all demand nodes because the storm affects a
      broad area without a single coastal strike.
    """
    return list(landfall) if landfall else list(all_demand_nodes)


# -- Public API -------------------------------------------------------------


def build_rawls_instance(
    seed: int = DEFAULT_SEED,
    truck_capacity_per_arc: float = 2_500.0,
    air_capacity_per_arc: float = 800.0,
    truck_unit_cost: float = 0.25,
    air_unit_cost: float = 1.60,
    airlift_available_all_arcs: bool = True,
) -> dict[str, Any]:
    """Construct the Rawls Gulf Coast instance in the shape expected by the
    Gurobi model. All keyword arguments carry reconstructed values that do not
    appear in the source paper and can be adjusted for sensitivity analyses.
    """
    del seed  # deterministic layout; seed retained for API compatibility

    nodes = list(NODES)
    demand_nodes = list(nodes)
    facilities = list(CANDIDATE_HUBS)
    arcs = _build_arcs(nodes, ARC_DISTANCE_THRESHOLD)
    if not _is_connected(nodes, arcs):
        raise RuntimeError(
            "Rawls arc graph is not connected; increase ARC_DISTANCE_THRESHOLD"
        )
    if not all(f in nodes for f in facilities):
        raise ValueError("Candidate hubs must be a subset of the node set")

    commodities = list(COMMODITIES)
    periods = list(PERIODS)
    facility_sizes = list(FACILITY_SIZES)
    modes = list(TRANSPORT_MODES)
    tau_m = {"truck": 1, "air": 0}

    scenario_names = list(RAW_SCENARIO_PROBS.keys())
    probs_norm = _normalize_probs(RAW_SCENARIO_PROBS)
    prob_sum = sum(probs_norm.values())
    if not math.isclose(prob_sum, 1.0, abs_tol=1e-9):
        raise RuntimeError(f"Renormalized probabilities do not sum to 1 (got {prob_sum})")

    # per-node arc distances (used for cost scaling)
    arc_distance: dict[tuple[int, int], float] = {
        (i, j): _euclid(NODE_COORDINATES[i], NODE_COORDINATES[j]) for (i, j) in arcs
    }

    # facility opening cost F_il: fixed base plus a small location-dependent
    # premium proportional to distance from the coast (safer inland hubs are
    # slightly cheaper to open; coastal ones marginally more expensive).
    F_il: dict[str, float] = {}
    for i in facilities:
        _, y_coord = NODE_COORDINATES[i]
        coastal_penalty = max(0.0, 3.0 - y_coord) * 350.0
        for l in facility_sizes:
            F_il[f"{i},{l}"] = SIZE_FIXED_COST[l] + coastal_penalty

    # commodity economics
    b_k = {k: COMMODITY_ECONOMICS[k]["b"] for k in commodities}
    q_k = {k: COMMODITY_ECONOMICS[k]["q"] for k in commodities}
    e_k = {k: COMMODITY_ECONOMICS[k]["e"] for k in commodities}
    h_k = {k: COMMODITY_ECONOMICS[k]["h"] for k in commodities}
    rho_k = {k: COMMODITY_ECONOMICS[k]["rho"] for k in commodities}
    M_l = dict(SIZE_CAPACITY)

    d_ikts: dict[str, float] = {}
    gamma_iks: dict[str, float] = {}
    U_ijmts: dict[str, float] = {}
    c_ijmkts: dict[str, float] = {}

    for s in scenario_names:
        h_id, _ = RAW_SCENARIO_PROBS[s]
        hurricane = HURRICANES[h_id]
        category = int(hurricane["category"])
        landfall = list(hurricane["landfall"])
        unusable = {tuple(sorted(link)) for link in hurricane["unusable_links"]}

        targets = _demand_targets(landfall, demand_nodes)
        per_target = 1.0 / len(targets)

        aggregate = {"water": hurricane["water"], "food": hurricane["food"], "medicine": hurricane["medicine"]}

        # demand: zero everywhere by default; positive at target nodes only
        for i in nodes:
            for k in commodities:
                for t in periods:
                    d_ikts[f"{i},{k},{t},{s}"] = 0.0
        for i in targets:
            for k in commodities:
                per_period = _demand_split(aggregate[k] * per_target)
                for t in periods:
                    d_ikts[f"{i},{k},{t},{s}"] = per_period[t]

        # damage: gamma at candidate hubs (unused for non-hubs, but recorded
        # so the model can look up any facility node uniformly)
        for i in facilities:
            g = _damage_gamma(category, landfall, i)
            for k in commodities:
                gamma_iks[f"{i},{k},{s}"] = g

        # arc capacities and costs
        for (i, j) in arcs:
            dist = arc_distance[(i, j)]
            truck_cap = 0.0 if tuple(sorted((i, j))) in unusable else truck_capacity_per_arc
            air_cap = air_capacity_per_arc if airlift_available_all_arcs else 0.0
            for t in periods:
                U_ijmts[f"{i},{j},truck,{t},{s}"] = truck_cap
                U_ijmts[f"{i},{j},air,{t},{s}"] = air_cap
                for k in commodities:
                    c_ijmkts[f"{i},{j},{k},truck,{t},{s}"] = truck_unit_cost * dist
                    c_ijmkts[f"{i},{j},{k},air,{t},{s}"] = air_unit_cost * dist

    instance = {
        "meta": {
            "seed": DEFAULT_SEED,
            "description": (
                "Rawls & Turnquist (2010) Gulf Coast test instance with 30 nodes, "
                "15 single-hurricane scenarios, multi-period demand split, and "
                "an added air transport mode."
            ),
            "version": 3,
            "source": "Rawls and Turnquist (2010), Tables 3 and 4",
            "node_coordinates": {str(n): list(NODE_COORDINATES[n]) for n in nodes},
            "raw_scenario_probability_sum": round(
                sum(p for _, p in RAW_SCENARIO_PROBS.values()), 6
            ),
            "raw_scenario_probabilities": {s: p for s, (_, p) in RAW_SCENARIO_PROBS.items()},
            "period_weights": {str(t): PERIOD_WEIGHTS[t] for t in periods},
            "hurricanes": {
                str(h): {
                    "category": HURRICANES[h]["category"],
                    "landfall": HURRICANES[h]["landfall"],
                    "unusable_links": [list(pair) for pair in HURRICANES[h]["unusable_links"]],
                }
                for h in HURRICANES
            },
            "scenario_hurricane_map": {s: h for s, (h, _) in RAW_SCENARIO_PROBS.items()},
        },
        "sets": {
            "nodes": nodes,
            "facilities": facilities,
            "hubs": facilities,
            "demand_nodes": demand_nodes,
            "arcs": [list(a) for a in arcs],
            "commodities": commodities,
            "periods": periods,
            "scenarios": scenario_names,
            "facility_sizes": facility_sizes,
            "transport_modes": modes,
        },
        "parameters": {
            "p_s": probs_norm,
            "d_ikts": d_ikts,
            "gamma_iks": gamma_iks,
            "U_ijmts": U_ijmts,
            "c_ijmkts": c_ijmkts,
            "tau_m": tau_m,
            "F_il": F_il,
            "M_l": M_l,
            "b_k": b_k,
            "q_k": q_k,
            "e_k": e_k,
            "h_k": h_k,
            "rho_k": rho_k,
            "alpha": 0.95,
            "lambda": 0.0,
        },
    }
    return instance


def save_instance(instance: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(instance, fh, indent=2)
    return out


def load_instance(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    instance = build_rawls_instance()
    path = Path(__file__).resolve().parent / "rawls_instance.json"
    save_instance(instance, path)
    n_nodes = len(instance["sets"]["nodes"])
    n_arcs = len(instance["sets"]["arcs"])
    n_scen = len(instance["sets"]["scenarios"])
    print(
        f"Wrote {path} with {n_nodes} nodes, {n_arcs} arcs, "
        f"{n_scen} scenarios, {len(instance['sets']['facilities'])} candidate hubs."
    )


if __name__ == "__main__":
    main()
