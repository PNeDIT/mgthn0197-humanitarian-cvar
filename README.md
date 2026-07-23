# Risk-Averse Humanitarian Logistics

Seminar project for **MGTHN0197 — Optimization and Data Science in Operations Management** (TUM, Summer Term 2026).

Two-stage stochastic MILP for disaster relief pre-positioning and distribution. The model compares risk-neutral planning with a mean-CVaR risk-averse formulation on the Rawls and Turnquist (2010) Gulf Coast case study, with multi-period demand and truck plus air transport.

## Repository layout

```
src/data/rawls_instance.py          Builds the Rawls Gulf Coast instance
src/models/cvar_model.py            Gurobi MILP (mean-CVaR)
src/experiments/run_experiments.py  Solves RN vs RA and writes results/
tests/                              Unit and integration tests
docs/references/                    Source papers used in the seminar
submission/                         Submitted paper and presentation PDFs
results/                            Generated CSVs and figures (not tracked)
```

## Case study

The default instance follows Rawls and Turnquist (2010), Tables 3 and 4:

- 30 nodes, 11 candidate hubs
- 15 single-hurricane scenarios (probabilities renormalized to sum to 1)
- Damage rule: category 3 or higher destroys landfall stock, category 1 to 2 leave 50 percent
- Unusable roads set truck capacity to zero, air stays available
- Commodities: water, food, medicine
- Three periods with demand split 50 / 30 / 20
- Modes: truck (delay 1 period) and air (delay 0)

Details that go beyond the tabulated Rawls data are documented in `src/data/rawls_instance.py`.

## Setup

Run every command from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi needs an unrestricted Named-User Academic licence. The size-limited Online Course licence is too small for the full Rawls network.

1. Request a Named-User Academic licence at `portal.gurobi.com` while on a TUM or LRZ network (VPN is fine).
2. Install the Gurobi Optimizer, then run `grbgetkey <UUID>` and accept `~/gurobi.lic`.
3. Check the licence with:

```bash
python -c "import gurobipy as g; g.Env()"
```

You should see `Academic license`.

## Run

```bash
python -m src.data.rawls_instance
python -m src.experiments.run_experiments
pytest tests/ -v
```

What each step does:

1. `rawls_instance` writes `src/data/rawls_instance.json`.
2. `run_experiments` solves the risk-neutral policy (`λ = 0`), the risk-averse policy (`λ = 18`, `α = 0.95`), and the λ-sweep `{0, 5, 10, 15, 18, 22}`. Outputs land in `results/`.
3. `pytest` checks the instance, the model, the CSV tables, and the figures.

Main files written to `results/`:

- `experiment_summary.csv`
- `lambda_sweep.csv`
- `per_hurricane.csv`
- PNG figures (policy comparison, coverage plots, Pareto curve, response profile, network map, CDF of unmet demand)

## Headline result

With `λ = 18` (numbers regenerate from `results/experiment_summary.csv`):

| Metric                                  | Risk-neutral | Risk-averse |
|-----------------------------------------|--------------|-------------|
| Expected total cost                     |    1,443,496 |   2,489,297 |
| CVaR₀.₉₅ of unmet demand                |       87,858 |      12,855 |
| Expected coverage                       |        83.5% |       93.4% |
| Coverage in worst hurricane (h9, Cat 5) |        19.6% |       88.2% |
| Hubs opened                             |            2 |           4 |
| Solve time (seconds)                    |         10.8 |        13.4 |

## Submission PDFs

| File | Content |
|------|---------|
| `submission/Seminar_OaDS_in_OM_Paper.pdf` | Seminar paper |
| `submission/Seminar_OaDS_in_OM_Initial_Presentation.pdf` | Topic pitch |
| `submission/Seminar_OaDS_in_OM_Interim_Presentation.pdf` | Interim presentation |
| `submission/Seminar_OaDS_in_OM_Final_Presentation.pdf` | Final presentation |

## Team

Petar Nedyalkov · Saadet Zehra Danaci · Aliyah Zahra Lathifa
