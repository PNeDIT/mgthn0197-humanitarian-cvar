"""Two-stage humanitarian logistics MILP with CVaR and multi-mode transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gurobipy as gp
from gurobipy import GRB


@dataclass
class SolveResult:
    status: str
    objective: float
    expected_cost: float
    cvar_value: float
    cvar_cost: float
    expected_coverage: float
    coverage_by_scenario: dict[str, float]
    cost_by_scenario: dict[str, float]
    unmet_by_scenario: dict[str, float]
    coverage_by_period: dict[str, dict[int, float]] = field(default_factory=dict)
    mean_unmet_period: dict[str, float] = field(default_factory=dict)
    coverage_by_commodity: dict[str, dict[str, float]] = field(default_factory=dict)
    unmet_by_commodity: dict[str, dict[str, float]] = field(default_factory=dict)
    lambda_param: float = 0.0
    alpha: float = 0.95
    solve_time_sec: float = 0.0
    open_facilities: list[dict[str, Any]] = field(default_factory=list)
    preposition: dict[str, float] = field(default_factory=dict)


def _key(*parts: Any) -> str:
    return ",".join(str(p) for p in parts)


class HumanitarianCVaRModel:
    """MILP for mean-CVaR humanitarian logistics with multi-mode flows and delivery delays."""

    def __init__(self, instance: dict[str, Any], lambda_risk: float | None = None, alpha: float | None = None):
        self.data = instance
        self.sets = instance["sets"]
        self.params = instance["parameters"]

        self.nodes = self.sets["nodes"]
        self.facilities = self.sets["facilities"]
        self.demand_nodes = self.sets["demand_nodes"]
        self.arcs = [tuple(a) for a in self.sets["arcs"]]
        self.commodities = self.sets["commodities"]
        self.periods = self.sets["periods"]
        self.scenarios = self.sets["scenarios"]
        self.facility_sizes = self.sets["facility_sizes"]
        self.modes = self.sets.get("transport_modes", ["truck"])
        self.periods0 = [0] + self.periods
        self.tau = {m: int(self.params.get("tau_m", {}).get(m, 0)) for m in self.modes}

        self.p = {s: float(self.params["p_s"][s]) for s in self.scenarios}
        self.alpha = float(alpha if alpha is not None else self.params.get("alpha", 0.95))
        self.lambda_risk = float(lambda_risk if lambda_risk is not None else self.params.get("lambda", 0.0))

        self.model: gp.Model | None = None
        self.vars: dict[str, Any] = {}

    def _d(self, i: int, k: str, t: int, s: str) -> float:
        return float(self.params["d_ikts"].get(_key(i, k, t, s), 0.0))

    def _gamma(self, i: int, k: str, s: str) -> float:
        return float(self.params["gamma_iks"][_key(i, k, s)])

    def _U(self, i: int, j: int, m: str, t: int, s: str) -> float:
        if "U_ijmts" in self.params:
            return float(self.params["U_ijmts"][_key(i, j, m, t, s)])
        return float(self.params["U_ijts"][_key(i, j, t, s)])

    def _c(self, i: int, j: int, k: str, m: str, t: int, s: str) -> float:
        if "c_ijmkts" in self.params:
            return float(self.params["c_ijmkts"][_key(i, j, k, m, t, s)])
        return float(self.params["c_ijkts"][_key(i, j, k, t, s)])

    def _F(self, i: int, l: str) -> float:
        return float(self.params["F_il"][_key(i, l)])

    def _inflow_expr(self, x, i: int, k: str, t: int, s: str):
        expr = gp.LinExpr()
        for (j, jj) in self.arcs:
            if jj != i:
                continue
            for m in self.modes:
                depart = t - self.tau[m]
                if depart in self.periods:
                    expr += x[j, i, k, depart, m, s]
        return expr

    def _outflow_expr(self, x, i: int, k: str, t: int, s: str):
        return gp.quicksum(x[i, j, k, t, m, s] for (ii, j) in self.arcs if ii == i for m in self.modes)

    def build(self) -> gp.Model:
        m = gp.Model("humanitarian_cvar")
        m.Params.OutputFlag = 0
        self.model = m

        y = m.addVars([(i, l) for i in self.facilities for l in self.facility_sizes], vtype=GRB.BINARY, name="y")
        r = m.addVars([(i, k) for i in self.facilities for k in self.commodities], lb=0.0, name="r")
        x = m.addVars(
            [
                (i, j, k, t, mode, s)
                for (i, j) in self.arcs
                for k in self.commodities
                for t in self.periods
                for mode in self.modes
                for s in self.scenarios
            ],
            lb=0.0,
            name="x",
        )
        I = m.addVars(
            [(i, k, t, s) for i in self.nodes for k in self.commodities for t in self.periods0 for s in self.scenarios],
            lb=0.0,
            name="I",
        )
        w = m.addVars(
            [(i, k, t, s) for i in self.nodes for k in self.commodities for t in self.periods for s in self.scenarios],
            lb=0.0,
            name="w",
        )
        Z = m.addVars(self.scenarios, lb=0.0, name="Z")
        S = m.addVars(self.scenarios, lb=0.0, name="S")
        eta = m.addVar(lb=-GRB.INFINITY, name="eta")
        xi = m.addVars(self.scenarios, lb=0.0, name="xi")

        b_k = self.params["b_k"]
        q_k = self.params["q_k"]
        e_k = self.params["e_k"]
        h_k = self.params["h_k"]
        rho_k = self.params["rho_k"]
        M_l = self.params["M_l"]

        C_first = gp.quicksum(self._F(i, l) * y[i, l] for i in self.facilities for l in self.facility_sizes) + gp.quicksum(
            q_k[k] * r[i, k] for i in self.facilities for k in self.commodities
        )

        for s in self.scenarios:
            Q_s = (
                gp.quicksum(
                    self._c(i, j, k, mode, t, s) * x[i, j, k, t, mode, s]
                    for (i, j) in self.arcs
                    for k in self.commodities
                    for t in self.periods
                    for mode in self.modes
                )
                + gp.quicksum(h_k[k] * I[i, k, t, s] for i in self.facilities for k in self.commodities for t in self.periods)
                + gp.quicksum(rho_k[k] * w[i, k, t, s] for i in self.demand_nodes for k in self.commodities for t in self.periods)
            )
            m.addConstr(Z[s] == C_first + Q_s, name=f"scenario_cost_{s}")
            m.addConstr(
                S[s]
                == gp.quicksum(w[i, k, t, s] for i in self.demand_nodes for k in self.commodities for t in self.periods),
                name=f"scenario_unmet_{s}",
            )

        for i in self.facilities:
            m.addConstr(
                gp.quicksum(b_k[k] * r[i, k] for k in self.commodities) <= gp.quicksum(M_l[l] * y[i, l] for l in self.facility_sizes),
                name=f"preposition_cap_{i}",
            )
            m.addConstr(gp.quicksum(y[i, l] for l in self.facility_sizes) <= 1, name=f"one_facility_{i}")

        non_facilities = [i for i in self.nodes if i not in self.facilities]
        non_demand = [i for i in self.nodes if i not in self.demand_nodes]

        for i in non_facilities:
            for k in self.commodities:
                for t in self.periods0:
                    for s in self.scenarios:
                        m.addConstr(I[i, k, t, s] == 0, name=f"nonfac_inv_{i}_{k}_{t}_{s}")

        for i in self.facilities:
            for t in self.periods:
                for s in self.scenarios:
                    m.addConstr(
                        gp.quicksum(b_k[kk] * I[i, kk, t, s] for kk in self.commodities)
                        <= gp.quicksum(M_l[l] * y[i, l] for l in self.facility_sizes),
                        name=f"inv_cap_{i}_{t}_{s}",
                    )

        for i in self.facilities:
            for k in self.commodities:
                for s in self.scenarios:
                    m.addConstr(I[i, k, 0, s] == self._gamma(i, k, s) * r[i, k], name=f"init_inv_{i}_{k}_{s}")

        for i in self.nodes:
            for k in self.commodities:
                for t in self.periods:
                    for s in self.scenarios:
                        m.addConstr(
                            I[i, k, t, s]
                            == I[i, k, t - 1, s]
                            + self._inflow_expr(x, i, k, t, s)
                            - self._outflow_expr(x, i, k, t, s)
                            - self._d(i, k, t, s)
                            + w[i, k, t, s],
                            name=f"balance_{i}_{k}_{t}_{s}",
                        )

        for i in self.demand_nodes:
            for k in self.commodities:
                for t in self.periods:
                    for s in self.scenarios:
                        m.addConstr(w[i, k, t, s] <= self._d(i, k, t, s), name=f"shortage_ub_{i}_{k}_{t}_{s}")

        for i in non_demand:
            for k in self.commodities:
                for t in self.periods:
                    for s in self.scenarios:
                        m.addConstr(w[i, k, t, s] == 0, name=f"shortage_zero_{i}_{k}_{t}_{s}")

        for (i, j) in self.arcs:
            for mode in self.modes:
                for t in self.periods:
                    for s in self.scenarios:
                        m.addConstr(
                            gp.quicksum(e_k[k] * x[i, j, k, t, mode, s] for k in self.commodities) <= self._U(i, j, mode, t, s),
                            name=f"transport_{i}_{j}_{mode}_{t}_{s}",
                        )

        for s in self.scenarios:
            m.addConstr(xi[s] >= S[s] - eta, name=f"cvar_excess_{s}")

        expected_cost_expr = gp.quicksum(self.p[s] * Z[s] for s in self.scenarios)
        cvar_shortage_expr = eta + (1.0 / (1.0 - self.alpha)) * gp.quicksum(self.p[s] * xi[s] for s in self.scenarios)
        m.setObjective(expected_cost_expr + self.lambda_risk * cvar_shortage_expr, GRB.MINIMIZE)

        self.vars = {"y": y, "r": r, "x": x, "I": I, "w": w, "Z": Z, "S": S, "eta": eta, "xi": xi}
        return m

    def solve(self, time_limit: float | None = None) -> SolveResult:
        if self.model is None:
            self.build()
        assert self.model is not None

        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit

        self.model.optimize()
        status = self._status_name(self.model.Status)
        if self.model.SolCount == 0:
            raise RuntimeError(f"Model not solved to a feasible solution (status={status})")

        y, r, w, Z, S, eta, xi = (
            self.vars["y"],
            self.vars["r"],
            self.vars["w"],
            self.vars["Z"],
            self.vars["S"],
            self.vars["eta"],
            self.vars["xi"],
        )

        cost_by_scenario = {s: float(Z[s].X) for s in self.scenarios}
        expected_cost = sum(self.p[s] * cost_by_scenario[s] for s in self.scenarios)
        cvar_shortage = float(eta.X) + (1.0 / (1.0 - self.alpha)) * sum(self.p[s] * float(xi[s].X) for s in self.scenarios)
        cvar_cost = self._empirical_cvar(cost_by_scenario)

        coverage_by_scenario: dict[str, float] = {}
        unmet_by_scenario: dict[str, float] = {}
        coverage_by_period: dict[str, dict[int, float]] = {}
        mean_unmet_period: dict[str, float] = {}
        coverage_by_commodity: dict[str, dict[str, float]] = {}
        unmet_by_commodity: dict[str, dict[str, float]] = {}

        for s in self.scenarios:
            total_d = sum(self._d(i, k, t, s) for i in self.demand_nodes for k in self.commodities for t in self.periods)
            total_w = sum(float(w[i, k, t, s].X) for i in self.demand_nodes for k in self.commodities for t in self.periods)
            unmet_by_scenario[s] = total_w
            coverage_by_scenario[s] = 1.0 - (total_w / total_d if total_d > 0 else 0.0)

            by_period: dict[int, float] = {}
            weighted = 0.0
            for t in self.periods:
                d_t = sum(self._d(i, k, t, s) for i in self.demand_nodes for k in self.commodities)
                w_t = sum(float(w[i, k, t, s].X) for i in self.demand_nodes for k in self.commodities)
                by_period[t] = 1.0 - (w_t / d_t if d_t > 0 else 0.0)
                weighted += t * w_t
            coverage_by_period[s] = by_period
            mean_unmet_period[s] = weighted / total_w if total_w > 1e-9 else 0.0

            cov_k: dict[str, float] = {}
            unmet_k: dict[str, float] = {}
            for k in self.commodities:
                d_k = sum(self._d(i, k, t, s) for i in self.demand_nodes for t in self.periods)
                w_k = sum(float(w[i, k, t, s].X) for i in self.demand_nodes for t in self.periods)
                unmet_k[k] = w_k
                cov_k[k] = 1.0 - (w_k / d_k if d_k > 0 else 0.0)
            coverage_by_commodity[s] = cov_k
            unmet_by_commodity[s] = unmet_k

        expected_coverage = sum(self.p[s] * coverage_by_scenario[s] for s in self.scenarios)

        open_facilities = []
        for i in self.facilities:
            for l in self.facility_sizes:
                if y[i, l].X > 0.5:
                    open_facilities.append({"node": i, "size": l})

        preposition = {f"{i},{k}": float(r[i, k].X) for i in self.facilities for k in self.commodities if r[i, k].X > 1e-6}

        return SolveResult(
            status=status,
            objective=float(self.model.ObjVal),
            expected_cost=expected_cost,
            cvar_value=cvar_shortage,
            cvar_cost=cvar_cost,
            expected_coverage=expected_coverage,
            coverage_by_scenario=coverage_by_scenario,
            cost_by_scenario=cost_by_scenario,
            unmet_by_scenario=unmet_by_scenario,
            coverage_by_period=coverage_by_period,
            mean_unmet_period=mean_unmet_period,
            coverage_by_commodity=coverage_by_commodity,
            unmet_by_commodity=unmet_by_commodity,
            lambda_param=self.lambda_risk,
            alpha=self.alpha,
            solve_time_sec=float(self.model.Runtime),
            open_facilities=open_facilities,
            preposition=preposition,
        )

    @staticmethod
    def _empirical_cvar(values: dict[str, float], alpha: float = 0.95) -> float:
        """Tail average of a discrete scenario distribution."""
        pairs = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
        tail_weight = 1.0 - alpha
        cumulative = 0.0
        weighted = 0.0
        for _, val in pairs:
            take = min(tail_weight - cumulative, 1.0 / len(values))
            if take <= 0:
                break
            weighted += take * val
            cumulative += take
        return weighted / tail_weight if tail_weight > 0 else pairs[0][1]

    @staticmethod
    def _status_name(code: int) -> str:
        mapping = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.SUBOPTIMAL: "suboptimal"}
        return mapping.get(code, f"status_{code}")


def solve_instance(
    instance: dict[str, Any],
    lambda_risk: float = 0.0,
    alpha: float = 0.95,
    time_limit: float | None = None,
) -> SolveResult:
    model = HumanitarianCVaRModel(instance, lambda_risk=lambda_risk, alpha=alpha)
    return model.solve(time_limit=time_limit)
