# football-forecast

From-first-principles forecasting of football matches: **scoreline distributions**,
**1X2 probabilities**, and **expected fouls / cards / corners**. Built as a
self-study project where every modelling choice is understood and justified.

National teams first (results-only data), then leagues (where event stats exist).

## Architecture in one picture

```
        PC (amd64, this machine)                         Pi 5 (arm64, 8 GB)
   ┌──────────────────────────────┐                ┌────────────────────────┐
   │ pipelines: ingest → train →  │   artifact     │ app/ : FastAPI + HTMX   │
   │ backtest → forecast          │ ── store ────▶ │ dashboard (read-only)   │
   │ (PyMC, LightGBM, Dixon–Coles)│  parquet+SQLite│ behind shared Caddy      │
   └──────────────────────────────┘                └────────────────────────┘
   heavy compute lives here                          only displays forecasts
```

Heavy compute (training, MCMC, backtesting) **only** runs on the PC. The Pi
serves a lightweight dashboard that reads precomputed forecasts. Full rationale:
[`docs/architecture.md`](docs/architecture.md).

## Documentation

| Doc | Purpose |
|-----|---------|
| [`roadmap.md`](roadmap.md) | The **research** plan — what to learn and in what order (the *why*) |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | The **dev** plan — modules, contracts, build sequence, acceptance criteria (the *how*) |
| [`CLAUDE.md`](CLAUDE.md) | Session primer for AI-assisted work; conventions at a glance |
| [`docs/models-explained.md`](docs/models-explained.md) | **Plain-language + math walkthrough of every model** — read this to understand the machinery |
| [`docs/architecture.md`](docs/architecture.md) | PC↔Pi compute split, artifact store, app design |
| [`docs/project-structure.md`](docs/project-structure.md) | Directory map and where things go |
| [`docs/data.md`](docs/data.md) | Data sources, target→source table, schemas, leakage hazards |
| [`docs/modeling-guidelines.md`](docs/modeling-guidelines.md) | Model families, backtesting rules, eval metrics |
| [`docs/workflow.md`](docs/workflow.md) | The standard pipeline + how to run things |
| [`docs/decisions.md`](docs/decisions.md) | Running log of modelling decisions (ADR-style) |
| [`deploy/DEPLOY.md`](deploy/DEPLOY.md) | Deploying the dashboard to the Pi homelab |

## Quick start (development, on the PC)

```bash
# create + activate a virtualenv, then:
pip install -e ".[dev]"          # editable install of the library + dev tools

# run the pipeline stages (stubs for now):
python -m pipelines.ingest       # fetch + normalize source data
python -m pipelines.train        # fit models on the PC
python -m pipelines.backtest     # walk-forward evaluation (RPS, log loss, calibration)
python -m pipelines.forecast     # write forecasts to the artifact store

# run the dashboard locally:
uvicorn app.main:app --reload    # http://localhost:8000
```

## Status

Phase 0 — scaffolding complete; no model code yet. See `roadmap.md` Part 7 for
the phased build plan.
