"""Compare risk-neutral and risk-averse policies on the Rawls case study.

The experiment layer solves the extended MILP for a grid of risk-aversion
weights, exports metric tables, and produces figures for the paper and the
final presentation. All numbers in the paper are traceable to the CSV files
this module writes into ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.rawls_instance import build_rawls_instance, load_instance, save_instance
from src.models.cvar_model import SolveResult, solve_instance

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
DEFAULT_INSTANCE = ROOT / "src" / "data" / "rawls_instance.json"

# Calibrated headline weight from the lambda sweep below.
LAMBDA_RISK_AVERSE = 18.0
LAMBDA_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 18.0, 22.0)
ALPHA = 0.95

RN_COLOR = "#2E6DA4"
RA_COLOR = "#D97B2C"
RN_LABEL = "Risk-neutral"
RA_LABEL = "Risk-averse"

FIGURE_NAMES: tuple[str, ...] = (
    "policy_comparison",
    "coverage_by_scenario",
    "coverage_by_commodity",
    "lambda_pareto",
    "response_time_extreme",
    "network_map",
    "cdf_unmet_demand",
)


# -- utilities -------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label_bars(ax: plt.Axes, bars, labels: list[str], offset: float = 0.0) -> None:
    for bar, text in zip(bars, labels):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )


def _ensure_instance(path: Path) -> dict:
    """Load the instance from disk, regenerating if it is stale or missing."""
    if not path.exists():
        instance = build_rawls_instance()
        save_instance(instance, path)
        return instance
    inst = load_instance(path)
    if inst.get("meta", {}).get("version", 1) < 3:
        instance = build_rawls_instance()
        save_instance(instance, path)
        return instance
    return inst


def _worst_scenario(coverage_by_scenario: dict[str, float]) -> str:
    return min(coverage_by_scenario.items(), key=lambda kv: kv[1])[0]


def _result_row(label: str, result: SolveResult) -> dict:
    row = {
        "policy": label,
        "lambda": result.lambda_param,
        "alpha": result.alpha,
        "objective": result.objective,
        "expected_cost": result.expected_cost,
        "cvar_shortage": result.cvar_value,
        "cvar_cost": result.cvar_cost,
        "expected_coverage": result.expected_coverage,
        "status": result.status,
        "n_hubs_opened": len(result.open_facilities),
        "solve_time_sec": result.solve_time_sec,
    }
    for s, cov in result.coverage_by_scenario.items():
        row[f"coverage_{s}"] = cov
    for s, cost in result.cost_by_scenario.items():
        row[f"cost_{s}"] = cost
    for s, unmet in result.unmet_by_scenario.items():
        row[f"unmet_{s}"] = unmet
    for s, delay in result.mean_unmet_period.items():
        row[f"response_delay_{s}"] = delay
    for s, by_t in result.coverage_by_period.items():
        for t, cov in by_t.items():
            row[f"coverage_{s}_t{t}"] = cov
    for s, by_k in result.coverage_by_commodity.items():
        for k, cov in by_k.items():
            row[f"coverage_{s}_{k}"] = cov
    return row


# -- solve pipeline --------------------------------------------------------


def run_comparison(
    instance_path: Path = DEFAULT_INSTANCE,
    lambda_risk_averse: float = LAMBDA_RISK_AVERSE,
    alpha: float = ALPHA,
) -> tuple[pd.DataFrame, SolveResult, SolveResult, dict]:
    instance = _ensure_instance(instance_path)
    rn = solve_instance(instance, lambda_risk=0.0, alpha=alpha)
    ra = solve_instance(instance, lambda_risk=lambda_risk_averse, alpha=alpha)
    df = pd.DataFrame([_result_row("risk_neutral", rn), _result_row("risk_averse", ra)])
    return df, rn, ra, instance


def run_lambda_sweep(
    instance_path: Path = DEFAULT_INSTANCE,
    alpha: float = ALPHA,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> pd.DataFrame:
    instance = _ensure_instance(instance_path)
    rows = []
    for lam in grid:
        result = solve_instance(instance, lambda_risk=float(lam), alpha=alpha)
        rows.append(_result_row(f"lambda_{lam}", result))
    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    rn: SolveResult,
    ra: SolveResult,
    instance: dict,
    out_dir: Path = RESULTS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "experiment_summary.csv", index=False)
    sweep.to_csv(out_dir / "lambda_sweep.csv", index=False)
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(df.to_dict(orient="records"), fh, indent=2)
    with (out_dir / "lambda_sweep.json").open("w", encoding="utf-8") as fh:
        json.dump(sweep.to_dict(orient="records"), fh, indent=2)

    per_hurricane = _per_hurricane_table(rn, ra, instance)
    per_hurricane.to_csv(out_dir / "per_hurricane.csv", index=False)

    _plot_policy_comparison(rn, ra, instance, out_dir)
    _plot_coverage_by_scenario(rn, ra, instance, out_dir)
    _plot_coverage_by_commodity(rn, ra, instance, out_dir)
    _plot_lambda_pareto(sweep, out_dir)
    _plot_response_time(rn, ra, instance, out_dir)
    _plot_network_map(instance, rn, ra, out_dir)
    _plot_cdf_unmet(rn, ra, instance, out_dir)


# -- derived tables --------------------------------------------------------


def _per_hurricane_table(rn: SolveResult, ra: SolveResult, instance: dict) -> pd.DataFrame:
    meta = instance.get("meta", {})
    scen_to_h = meta.get("scenario_hurricane_map", {})
    hurricanes = meta.get("hurricanes", {})
    probs = instance["parameters"]["p_s"]

    rows = []
    for s in instance["sets"]["scenarios"]:
        h_id = scen_to_h.get(s, "")
        h_info = hurricanes.get(str(h_id), {})
        row = {
            "scenario": s,
            "hurricane": h_id,
            "category": h_info.get("category"),
            "landfall": ",".join(str(x) for x in h_info.get("landfall", [])),
            "probability": probs[s],
            "rn_coverage": rn.coverage_by_scenario[s],
            "ra_coverage": ra.coverage_by_scenario[s],
            "rn_unmet": rn.unmet_by_scenario[s],
            "ra_unmet": ra.unmet_by_scenario[s],
            "rn_cost": rn.cost_by_scenario[s],
            "ra_cost": ra.cost_by_scenario[s],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


# -- figures ---------------------------------------------------------------


def _plot_policy_comparison(rn: SolveResult, ra: SolveResult, instance: dict, out_dir: Path) -> None:
    worst_s = _worst_scenario(rn.coverage_by_scenario)
    labels = [RN_LABEL, RA_LABEL]
    x = np.arange(2)
    width = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))

    panels = [
        (axes[0], [rn.expected_cost, ra.expected_cost], "Expected total cost", lambda v: f"{v:,.0f}"),
        (axes[1], [rn.cvar_value, ra.cvar_value], r"CVaR$_{0.95}$ of unmet demand", lambda v: f"{v:,.0f}"),
        (axes[2], [rn.coverage_by_scenario[worst_s] * 100, ra.coverage_by_scenario[worst_s] * 100],
         f"Coverage in worst hurricane ({worst_s}) [%]", lambda v: f"{v:.1f}%"),
        (axes[3], [rn.unmet_by_scenario[worst_s], ra.unmet_by_scenario[worst_s]],
         f"Unmet demand in {worst_s} (units)", lambda v: f"{v:,.0f}"),
    ]
    for ax, values, title, fmt in panels:
        bars = ax.bar(x, values, width=width, color=[RN_COLOR, RA_COLOR], edgecolor="white", linewidth=0.8)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        _style_axes(ax)
        ymin, ymax = min(values), max(values)
        pad = (ymax - ymin) * 0.24 if ymax > ymin else max(ymax * 0.08, 1.0)
        ax.set_ylim(max(0, ymin - pad * 0.25), ymax + pad)
        _label_bars(ax, bars, [fmt(v) for v in values])

    fig.suptitle(
        f"Policy comparison (λ = 0 vs λ = {int(ra.lambda_param)}) — Rawls Gulf Coast",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "policy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_by_scenario(rn: SolveResult, ra: SolveResult, instance: dict, out_dir: Path) -> None:
    scenarios = list(rn.coverage_by_scenario.keys())
    probs = instance["parameters"]["p_s"]
    order = sorted(scenarios, key=lambda s: rn.coverage_by_scenario[s])
    rn_vals = [rn.coverage_by_scenario[s] * 100 for s in order]
    ra_vals = [ra.coverage_by_scenario[s] * 100 for s in order]
    prob_pct = [probs[s] * 100 for s in order]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(order))
    width = 0.38
    bars_rn = ax.bar(x - width / 2, rn_vals, width, label=RN_LABEL, color=RN_COLOR, edgecolor="white")
    bars_ra = ax.bar(x + width / 2, ra_vals, width, label=RA_LABEL, color=RA_COLOR, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({p:.1f}%)" for s, p in zip(order, prob_pct)], fontsize=8)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Coverage per Rawls hurricane scenario (sorted by RN coverage; probability shown below)")
    ax.legend(loc="upper left", frameon=False)
    _style_axes(ax)
    _label_bars(ax, bars_rn, [f"{v:.0f}" for v in rn_vals])
    _label_bars(ax, bars_ra, [f"{v:.0f}" for v in ra_vals])
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_scenario.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_by_commodity(rn: SolveResult, ra: SolveResult, instance: dict, out_dir: Path) -> None:
    commodities = instance["sets"]["commodities"]
    worst_s = _worst_scenario(rn.coverage_by_scenario)

    rn_vals = [rn.coverage_by_commodity[worst_s][k] * 100 for k in commodities]
    ra_vals = [ra.coverage_by_commodity[worst_s][k] * 100 for k in commodities]

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(commodities))
    width = 0.38
    b_rn = ax.bar(x - width / 2, rn_vals, width, label=RN_LABEL, color=RN_COLOR, edgecolor="white")
    b_ra = ax.bar(x + width / 2, ra_vals, width, label=RA_LABEL, color=RA_COLOR, edgecolor="white")
    ax.set_xticks(x, [k.capitalize() for k in commodities])
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 118)
    ax.set_title(f"Per-commodity coverage in worst hurricane ({worst_s})")
    ax.legend(frameon=False)
    _style_axes(ax)
    _label_bars(ax, b_rn, [f"{v:.1f}%" for v in rn_vals])
    _label_bars(ax, b_ra, [f"{v:.1f}%" for v in ra_vals])
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_by_commodity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_pareto(sweep: pd.DataFrame, out_dir: Path) -> None:
    sweep = sweep.sort_values("lambda").reset_index(drop=True)
    xs = sweep["expected_cost"].to_list()
    ys = sweep["cvar_shortage"].to_list()
    lams = sweep["lambda"].to_list()

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    ax.plot(xs, ys, "-", color=RN_COLOR, linewidth=1.6, zorder=2)
    ax.scatter(xs, ys, s=95, color=RA_COLOR, edgecolors="white", linewidths=1.0, zorder=3)
    offsets = [(10, 12), (10, -18), (-10, 14), (10, -18), (10, 14), (-10, -18)]
    for i, (lam, x, y) in enumerate(zip(lams, xs, ys)):
        ax.annotate(
            f"λ = {int(lam) if float(lam).is_integer() else lam}",
            (x, y),
            textcoords="offset points",
            xytext=offsets[i % len(offsets)],
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )
    ax.set_xlabel("Expected total cost")
    ax.set_ylabel(r"CVaR$_{0.95}$ of unmet demand (units)")
    ax.set_title(r"Cost–CVaR trade-off across $\lambda$")
    _style_axes(ax)
    xr = max(xs) - min(xs)
    yr = max(ys) - min(ys) if max(ys) > min(ys) else max(ys)
    ax.set_xlim(min(xs) - 0.05 * xr, max(xs) + 0.08 * xr)
    ax.set_ylim(min(ys) - 0.08 * yr - 200, max(ys) + 0.15 * yr + 200)

    table_data = [
        [
            f"{int(lam) if float(lam).is_integer() else lam}",
            f"{x:,.0f}",
            f"{y:,.0f}",
        ]
        for lam, x, y in zip(lams, xs, ys)
    ]
    table = ax_tbl.table(
        cellText=table_data,
        colLabels=["λ", "E[Z]", "CVaR(S)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.35)
    ax_tbl.set_title("Exact values", fontsize=10, pad=12)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(out_dir / "lambda_pareto.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_response_time(rn: SolveResult, ra: SolveResult, instance: dict, out_dir: Path) -> None:
    worst_s = _worst_scenario(rn.coverage_by_scenario)
    periods = sorted(rn.coverage_by_period[worst_s].keys())
    rn_vals = [rn.coverage_by_period[worst_s][t] * 100 for t in periods]
    ra_vals = [ra.coverage_by_period[worst_s][t] * 100 for t in periods]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(periods, rn_vals, "o-", label=RN_LABEL, color=RN_COLOR, linewidth=2, markersize=7)
    ax.plot(periods, ra_vals, "o-", label=RA_LABEL, color=RA_COLOR, linewidth=2, markersize=7)
    for t, v in zip(periods, rn_vals):
        ax.annotate(f"{v:.0f}%", (t, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=RN_COLOR)
    for t, v in zip(periods, ra_vals):
        ax.annotate(f"{v:.0f}%", (t, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=RA_COLOR)
    ax.set_xlabel("Period (1 = immediate, 3 = sustained)")
    ax.set_ylabel(f"Coverage in {worst_s} (%)")
    ax.set_title(f"Response profile in worst hurricane ({worst_s})")
    ax.set_xticks(periods)
    ymax = max(max(rn_vals), max(ra_vals))
    ax.set_ylim(0, min(118, ymax * 1.35 + 5))
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "response_time_extreme.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_network_map(instance: dict, rn: SolveResult, ra: SolveResult, out_dir: Path) -> None:
    meta = instance.get("meta", {})
    coords = {int(n): tuple(v) for n, v in meta.get("node_coordinates", {}).items()}
    if not coords:
        return
    facilities = set(instance["sets"]["facilities"])
    landfall_counts: dict[int, int] = {}
    for _h_id, info in meta.get("hurricanes", {}).items():
        for node in info.get("landfall", []):
            landfall_counts[node] = landfall_counts.get(node, 0) + 1

    rn_open = {f["node"] for f in rn.open_facilities}
    ra_open = {f["node"] for f in ra.open_facilities}
    both = rn_open & ra_open
    rn_only = rn_open - ra_open
    ra_only = ra_open - rn_open

    fig, ax = plt.subplots(figsize=(11, 5.2))

    for arc in instance["sets"]["arcs"]:
        i, j = arc
        if i in coords and j in coords:
            (x0, y0), (x1, y1) = coords[i], coords[j]
            ax.plot([x0, x1], [y0, y1], color="#DDDDDD", linewidth=0.6, zorder=1)

    other_x = [coords[n][0] for n in coords if n not in facilities and n not in landfall_counts]
    other_y = [coords[n][1] for n in coords if n not in facilities and n not in landfall_counts]
    ax.scatter(other_x, other_y, s=28, color="#7A7A7A", zorder=3, label="Demand node")

    lf_x = [coords[n][0] for n in landfall_counts]
    lf_y = [coords[n][1] for n in landfall_counts]
    lf_size = [40 + 60 * landfall_counts[n] for n in landfall_counts]
    ax.scatter(
        lf_x,
        lf_y,
        s=lf_size,
        color="#B03030",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="Hurricane landfall",
    )

    hub_x = [coords[n][0] for n in facilities]
    hub_y = [coords[n][1] for n in facilities]
    ax.scatter(
        hub_x,
        hub_y,
        s=90,
        marker="s",
        facecolors="none",
        edgecolors=RN_COLOR,
        linewidths=1.4,
        zorder=5,
        label="Candidate hub",
    )

    # RN-only: blue square slightly left of the node.
    for n in rn_only:
        x, y = coords[n]
        ax.scatter(x - 0.28, y, s=150, marker="s", color=RN_COLOR, edgecolors="white", linewidths=1.0, zorder=6)

    # RA-only: orange diamond slightly right of the node.
    for n in ra_only:
        x, y = coords[n]
        ax.scatter(x + 0.28, y, s=170, marker="D", color=RA_COLOR, edgecolors="white", linewidths=1.0, zorder=7)

    # Shared hubs: nested square + diamond so both policies stay visible.
    for n in both:
        x, y = coords[n]
        ax.scatter(x, y, s=210, marker="s", color=RN_COLOR, edgecolors="white", linewidths=1.0, zorder=6)
        ax.scatter(x, y, s=95, marker="D", color=RA_COLOR, edgecolors="white", linewidths=1.0, zorder=7)

    for n, (x, y) in coords.items():
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#333333")

    proxy_rn = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=RN_COLOR,
        markersize=10,
        label=f"Opened by RN only ({len(rn_only)})",
    )
    proxy_ra = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor=RA_COLOR,
        markersize=10,
        label=f"Opened by RA only ({len(ra_only)})",
    )
    proxy_both = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=RN_COLOR,
        markeredgecolor=RA_COLOR,
        markeredgewidth=2.0,
        markersize=10,
        label=f"Opened by both ({len(both)}, nested)",
    )
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [proxy_rn, proxy_ra, proxy_both],
        loc="lower right",
        fontsize=8,
        frameon=True,
    )

    ax.set_title("Rawls Gulf Coast network: landfall exposure and opened hubs")
    ax.set_xlabel("longitude proxy")
    ax.set_ylabel("distance from coast (proxy)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 6.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "network_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cdf_unmet(rn: SolveResult, ra: SolveResult, instance: dict, out_dir: Path) -> None:
    """Empirical CDF of scenario unmet demand for RN and RA policies."""
    probs = instance["parameters"]["p_s"]
    scenarios = list(rn.unmet_by_scenario.keys())

    def _cdf_points(unmet: dict[str, float]) -> tuple[list[float], list[float]]:
        pairs = sorted(((unmet[s], probs[s]) for s in scenarios), key=lambda kv: kv[0])
        xs: list[float] = []
        ys: list[float] = []
        cum = 0.0
        for val, p in pairs:
            cum += p
            xs.append(val)
            ys.append(cum)
        return xs, ys

    rn_x, rn_y = _cdf_points(rn.unmet_by_scenario)
    ra_x, ra_y = _cdf_points(ra.unmet_by_scenario)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(rn_x, [100 * y for y in rn_y], where="post", color=RN_COLOR, linewidth=2.2, label=RN_LABEL)
    ax.step(ra_x, [100 * y for y in ra_y], where="post", color=RA_COLOR, linewidth=2.2, label=RA_LABEL)
    ax.scatter(rn_x, [100 * y for y in rn_y], color=RN_COLOR, s=35, zorder=3)
    ax.scatter(ra_x, [100 * y for y in ra_y], color=RA_COLOR, s=35, zorder=3)
    ax.set_xlabel("Unmet demand (units)")
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_title("Empirical CDF of unmet demand across Rawls hurricane scenarios")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "cdf_unmet_demand.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -- entry point -----------------------------------------------------------


def main() -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200})
    df, rn, ra, instance = run_comparison()
    sweep = run_lambda_sweep()
    save_outputs(df, sweep, rn, ra, instance)
    print(df.to_string(index=False))
    print(f"\nHeadline λ = {LAMBDA_RISK_AVERSE}")
    print(f"Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
