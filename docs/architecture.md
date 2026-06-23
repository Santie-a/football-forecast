# Architecture

## Goals

1. Forecast scoreline / 1X2 / counts with models we understand from first principles.
2. Visualize forecasts on a small always-on dashboard reachable from the homelab.
3. Keep heavy compute off the Pi; keep the Pi simple, read-only, and robust.

## The compute split (the central decision)

| Concern | Where | Why |
|---------|-------|-----|
| Ingest, clean, EDA | **PC** | Pandas/notebook work, iterative, memory-hungry |
| Train Elo / Dixon–Coles | **PC** | Optimization over full history |
| Bayesian MCMC (PyMC) | **PC only** | Sampling is the heaviest step; arm64 Pi is too slow/limited |
| Gradient boosting | **PC** | Training is CPU-heavy |
| Walk-forward backtests | **PC** | Re-fits across many time windows |
| Generate forecasts for upcoming matches | **PC** | Runs the trained models, writes the store |
| **Serve the dashboard** | **Pi** | Read-only, light, always-on |
| Cheap live inference (scoreline matrix from saved λ params) | **Pi OK** | A Dixon–Coles matrix is a trivial calc — fine for "what-if" sliders |

Rule of thumb: **if it fits a model or samples a posterior, it runs on the PC.**
If it only reads numbers and draws charts, it runs on the Pi.

## Handoff: the artifact store (batch)

The PC and Pi are decoupled by a **batch artifact store** — no live request from
the Pi to the PC at view time. The PC produces; the Pi consumes.

**Layout** (`artifacts/`, gitignored, synced PC→Pi):

```
artifacts/
├── forecasts/
│   └── forecasts.sqlite      # one row per (match, model, market) with probabilities
├── models/
│   └── <model>-<date>.pkl    # serialized fitted models / posterior summaries
└── metrics/
    └── backtests.parquet     # model × split × metric (RPS, log loss, Brier, calibration)
```

**Why SQLite for forecasts:** the dashboard does point lookups and small filtered
queries ("forecasts for this fixture / this model"); SQLite is a single file,
trivial to sync, and needs no server. **Parquet** is used for bulk analytical
outputs (backtest tables) the PC re-reads for comparison.

**The store is the contract.** Its schema is defined in
`src/football_forecast/store/` and documented in
[`data.md`](data.md#artifact-store-schema). The app depends only on that schema,
never on the modelling code — so models can change freely as long as the store
schema holds.

**Sync mechanism (decide & log in `decisions.md`):** simplest first — the deploy
copies `artifacts/forecasts/forecasts.sqlite` to the Pi (scp/rsync over Tailscale,
or a bind-mounted path the deploy refreshes). A scheduled `rsync` can automate it
later. The Pi never writes to the store.

## The dashboard (Pi): FastAPI + HTMX

- **FastAPI** serves a few routes: fixtures list, a match detail page (scoreline
  heatmap, 1X2 bars, over/under, expected counts), and a model-comparison page
  (backtest metrics).
- **HTMX + server-rendered templates** (Jinja2) for interactivity without a JS
  build step — fragments swapped in on click. Charts via a light option (server
  rendered SVG, or a small client lib like Chart.js/uPlot from a CDN/static file).
- **Read-only**: opens the SQLite store read-only; holds no training code and no
  heavy deps (no PyMC/LightGBM in the app's requirements — keep its image small
  for arm64).
- Packaged as its own container; behind the shared Caddy at
  `forecast.homeserver.internal`. See [`deploy/DEPLOY.md`](../deploy/DEPLOY.md).

## Fixtures & the forecast queue

Upcoming matches and user-entered results are managed in a **separate, writable**
`fixtures.sqlite` (DAL: `store/fixtures.py`, stdlib-only) — distinct from the
read-only forecasts store so the one-way PC→Pi sync never clobbers Pi-side writes.

```
add fixture / result  ──▶  fixtures.sqlite (forecast = NULL ⇒ "queued")
                                  │
                  drain (cheap inference, fitted model)
                                  │           Pi: synced model pickle  |  PC: fit fresh
                                  ▼
                       fixture row gets its forecast bundle  ⇒ "forecast"
```

- **Adding a fixture/result** (dashboard form or `pipelines`) only *records* it —
  no compute. New fixtures are "queued" (no forecast yet).
- **Draining the queue** (`fixtures_queue.drain`, via `pipelines.process_queue`)
  computes the market bundle with an already-fitted model. A Dixon–Coles matrix is
  cheap, so this is allowed on the Pi (loading a synced model pickle); the PC can
  also do it. **No training happens in the queue** — folding new results back into
  the model is a separate PC batch job.
- **Seeding:** `pipelines.wc2026` loads the real WC2026 group matches from the
  results data with pre-tournament forecasts (fit asof the tournament start).

This is a deliberate, bounded **reverse channel** (Pi records requests; compute
stays cheap or on the PC) — it does not turn the Pi into a training box (001).

### Optional: on-demand heavy compute (future, not now)

If a "recompute this with the Bayesian model" button is ever needed live, the Pi
app would call a small service on the PC by **Tailscale name** (the PC must be
on). Deferred — the batch store covers all current needs. Logged as an open
option in `decisions.md`.

## Dependency boundaries

```
app/            depends on → store schema only (sqlite3, fastapi, jinja2, htmx)
pipelines/      depends on → src/football_forecast/* (full DS stack)
src/football_forecast/  the only place modelling logic lives
research/       depends on → src/football_forecast/* (notebooks call the library)
```

Two dependency groups in `pyproject.toml`: the **core/app** deps (light) and the
**research/training** deps (PyMC, LightGBM, statsmodels, jupyter). The Pi installs
only the light group.
