# Project structure

Where everything lives and the rule for deciding where new code goes.

```
football-forecast/
├── roadmap.md                  research/build plan (read-only reference)
├── CLAUDE.md                   session primer + conventions
├── README.md                   project overview + quick start
├── pyproject.toml              package metadata + dependency groups
├── .gitignore
├── .env.example                template for secrets/config (real .env is gitignored)
│
├── docs/                       all design + guideline docs (this folder)
│
├── src/football_forecast/      THE LIBRARY — the only place reusable logic lives
│   ├── data/                   ingest + clean + normalize (team-name joins, dates, encodings)
│   ├── features/               feature engineering (Elo diff, rolling form, rest days, …)
│   ├── models/                 elo, dixon_coles, bayesian (PyMC), boosting, count models
│   ├── forecast/               KEYSTONE: scoreline matrix → 1X2 / over-under / correct-score
│   ├── eval/                   RPS, log loss, Brier, calibration, walk-forward backtester
│   └── store/                  read/write the artifact store (forecasts.sqlite, parquet)
│
├── pipelines/                  CLI entrypoints run on the PC; they CALL the library
│   ├── ingest.py               fetch + normalize source data → data/
│   ├── train.py                fit models → artifacts/models/
│   ├── backtest.py             walk-forward eval → artifacts/metrics/
│   └── forecast.py             produce upcoming-match forecasts → artifacts/forecasts/
│
├── research/                   exploratory work
│   ├── notebooks/              EDA, experiments (not imported by anything)
│   └── notes/                  the ½–1 page write-ups roadmap.md requires per topic
│
├── app/                        FastAPI + HTMX dashboard (deploys to the Pi; read-only)
│   ├── main.py                 routes
│   ├── templates/              Jinja2 + HTMX fragments
│   └── static/                 css, small JS/charts
│
├── data/                       datasets (GITIGNORED)
│   ├── raw/                    exactly as downloaded — never edited
│   ├── interim/                partially cleaned
│   └── processed/              model-ready
│
├── artifacts/                  model outputs (GITIGNORED; synced PC→Pi)
│   ├── forecasts/forecasts.sqlite
│   ├── models/
│   └── metrics/
│
├── deploy/                     Pi deployment (DEPLOY.md, compose, Caddy snippet)
└── tests/                      pytest (mirrors src/ layout)
```

## The "where does this go?" rule

- **Reusable logic** (anything a second caller might want) → `src/football_forecast/`.
- **A thing you run** (a stage, a job) → `pipelines/` — thin; it parses args,
  calls the library, writes an artifact.
- **Throwaway / exploratory** → `research/notebooks/`. If it proves useful,
  *promote* it into the library, don't leave it in a notebook.
- **Anything user-facing/visual** → `app/`. The app reads the store; it must not
  import modelling code or heavy deps.
- **A write-up that explains your understanding** → `research/notes/` (these
  satisfy roadmap.md's "explain it without notes" requirement).

## Naming conventions

- Python: `snake_case` modules and functions, `PascalCase` classes.
- One model family per module in `models/` (`elo.py`, `dixon_coles.py`,
  `bayesian.py`, `boosting.py`, `counts.py`).
- Tests mirror source paths: `src/.../models/elo.py` ↔ `tests/models/test_elo.py`.
- Research notes: `research/notes/<part>-<topic>.md` (e.g. `04-keystone-scoreline-matrix.md`).
