# Towards Welfare-Oriented Recommendations in Activity-Travel Behavior

Code and simulation framework accompanying the paper:

> **Towards welfare-oriented recommendations in activity-travel behavior**\
> Ekin Ugurel and Takahiro Yabe (Department of Technology Management and Innovation, New York University)\
> *Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)*, Minneapolis, MN, USA, September 28 – October 2, 2026.

**Corresponding author:** Ekin Ugurel ([eu2158@nyu.edu](mailto:eu2158@nyu.edu))

## Overview

Mainstream recommender systems (RS) rank alternatives using heuristics like popularity, proximity, and collaborative filtering, but lack a principled account of *user welfare* — whether accepting a recommendation actually leaves the user better off. This is especially costly in activity-travel settings, where users incur unrecoverable travel costs (time, energy, money) regardless of how satisfying the destination turns out to be.

This repository implements a **welfare-oriented framework for activity recommendation** that evaluates suggestions in terms of *net utility* (experienced benefit minus generalized travel cost), and formalizes two operational decision criteria:

- **Positive Utility Probability (PUP):** recommend only when the probability of non-negative net utility exceeds a threshold α.
- **Regret Minimization (RM):** recommend only when expected regret relative to the user's best organic (self-selected) alternative falls below a tolerance ε.

Both criteria are evaluated inside an **agent-based simulation (ABM)** in which heterogeneous synthetic travelers interact with multiple recommender systems over time in a spatial environment with realistic travel costs, congestion, and behavioral feedback loops — enabling controlled counterfactual comparisons against No-RS, Standard-RS, and perfect-knowledge Oracle baselines.

## Repository Structure

```
.
├── paper_figures_and_tables.ipynb   # Reproduces every figure and table in the paper
├── simulation.py                    # Simulation orchestrator (main entry point of the ABM)
├── agent.py                         # Traveler agent: preferences, decision-making, feedback
├── city.py                          # Spatial grid environment (venues, distances, congestion)
├── datastructures.py                # Core shared data structures
├── params.py                        # Central registry of all tunable model parameters
├── recommender_systems.py           # Standard RS stack (Google Maps-style prominence/relevance/proximity replica)
├── welfare_layer.py                 # Welfare-aware layer: PUP & RM criteria, travel-cost estimator, orchestrator
├── oracle_rs.py                     # Oracle RS: perfect-knowledge upper bound on achievable welfare
├── experiment_harness.py            # Matched-seed paired-comparison harness + welfare metrics (ORC, Gini, ε calibration)
├── utils.py                         # Small numeric helpers (softmax, clamp, ...)
├── data/
│   └── synthetic_personas_..._Seattle.csv   # Synthetic traveler personas with start locations
├── figs/                            # Pre-generated paper figures and result tables
└── requirements.txt                 # Python dependencies
```

## Dependency Flow

1. **`params.py`** defines every numeric assumption (utility weights, cost parameters, persona file path, etc.). Start here to inspect or modify the model.
2. **`city.py`** + **`datastructures.py`** build the spatial world; **`agent.py`** populates it with heterogeneous travelers instantiated from the personas in `data/`.
3. **`recommender_systems.py`** provides the Standard RS baseline; **`welfare_layer.py`** wraps any base RS with the PUP/RM welfare gate (`WelfareAwareOrchestrator` + `TravelCostEstimator`); **`oracle_rs.py`** provides the perfect-information upper bound.
4. **`simulation.py`** ties it all together: agents choose activities each tick, travel, experience utility, and feed satisfaction back to the RS.
5. **`experiment_harness.py`** runs matched-seed treatments across RS conditions and computes the paper's welfare metrics: welfare gain, over-recommendation cost (ORC), Gini coefficient, and RM ε calibration.
6. **`paper_figures_and_tables.ipynb`** drives everything above to reproduce the paper's results.

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd welfare_oriented_rec_sys
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `jupyter`.

## Reproducing the Paper's Results

All figures and tables are reproduced by running **`paper_figures_and_tables.ipynb`** top to bottom:

```bash
jupyter notebook paper_figures_and_tables.ipynb
```

The notebook is organized to mirror the paper:

| Section | Produces | Description |
|---|---|---|
| **0. Imports & shared configuration** | — | Simulation settings (250 agents, 18×18 city, 60 days, 5 matched seeds) and the six treatments: No RS, Standard RS, PUP-0.3, PUP-0.7, RM, Oracle. Runs and caches all treatment simulations. |
| **1. Sensitivity analysis** | Figures 2 & 3, Table 1 | PUP α sweep (α ∈ [0, 1]) and RM ε calibration/sweep, tracing the welfare–coverage frontier for each criterion; aggregate welfare comparison across treatments. |
| **2. Heterogeneity analysis** | Paradigm heatmap | Per-agent mean utility broken down by decision-making paradigm and treatment. |
| **3. Cost of over-recommendation** | ORC table | Share of agents improved vs. harmed by recommendations under each condition. |

Notes on replication:

- **Determinism:** all treatments use matched seeds (`SEEDS = [11, 13, 17, 19, 23]`), so re-running the notebook reproduces the paper's numbers exactly.
- **Runtime:** Section 0 runs 6 treatments × 5 seeds, and Section 1 adds the α/ε sweeps (cached runs are reused where possible). Expect the full notebook to take a while on a laptop; results are cached in memory so downstream sections are fast.
- **Outputs:** figures render inline in the notebook. The versions used in the paper, along with the ORC table (`orc_table.csv`) and numeric results (`latex_results.json`), are archived in `figs/`.

To experiment beyond the paper — different thresholds, city sizes, persona sets — edit the configuration cell at the top of the notebook or the corresponding entries in `params.py`.

## Citation

```bibtex
@inproceedings{ugurel2026welfare,
  author    = {Ugurel, Ekin and Yabe, Takahiro},
  title     = {Towards Welfare-Oriented Recommendations in Activity-Travel Behavior},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year      = {2026},
  address   = {Minneapolis, MN, USA},
  publisher = {ACM},
  doi       = {10.1145/nnnnnnn.nnnnnnn}  % TODO: update when proceedings are published
}
```

## Contact

Questions or issues: Ekin Ugurel — [eu2158@nyu.edu](mailto:eu2158@nyu.edu)
