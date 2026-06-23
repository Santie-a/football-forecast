# Workflow

The standard pipeline (roadmap.md Part 5) mapped onto this repo, plus how to run it.

## The pipeline, stage by stage

| Stage | Command | Reads | Writes |
|-------|---------|-------|--------|
| 1. Ingest | `python -m pipelines.ingest` | sources | `data/raw/`, `data/processed/` |
| 2. Clean/normalize | (inside ingest) | `data/raw/` | `data/interim/`, `data/processed/` |
| 3. EDA | `research/notebooks/` | `data/processed/` | `research/notes/` write-ups |
| 4. Features | (library, called by train) | `data/processed/` | feature frames |
| 5. Fit | `python -m pipelines.train` | features | `artifacts/models/` |
| 6. Derive forecasts | `python -m pipelines.forecast` | models | `artifacts/forecasts/forecasts.sqlite` |
| 7. Backtest | `python -m pipelines.backtest` | features, models | `artifacts/metrics/backtests.parquet` |
| 8. Evaluate/calibrate | (inside backtest) | — | metrics + calibration |
| 9. Compare + log | review + `docs/decisions.md` | metrics | a decision entry |

All of stages 1–8 run **on the PC**. The Pi only reads the artifacts produced.

> **Golden rule:** never shuffle time-series matches into random splits. Backtests
> are walk-forward, train-on-past only.

## Per-phase loop (from roadmap.md Part 7)

For each phase: resolve the relevant scope questions → acquire/confirm data → build
the model in `src/football_forecast/models/` → derive forecasts via the keystone →
backtest on time-ordered splits → compare to the previous phase → log the decision.
Keep the end-to-end pipeline runnable at all times, however simple the model.

## Running things (PC, development)

```bash
pip install -e ".[dev,research]"     # library + dev + training deps
python -m pipelines.ingest
python -m pipelines.train
python -m pipelines.backtest
python -m pipelines.forecast
uvicorn app.main:app --reload        # dashboard at http://localhost:8000
pytest                               # tests
```

## Publishing forecasts to the Pi

1. Run `pipelines.forecast` on the PC → refreshes `artifacts/forecasts/forecasts.sqlite`.
2. Sync that file to the Pi (see [`deploy/DEPLOY.md`](../deploy/DEPLOY.md)).
3. The dashboard picks up the new data on next request (read-only open).

## Session checklist for AI-assisted work

1. Read `CLAUDE.md`, then the current phase in `roadmap.md`.
2. Check `docs/decisions.md` for settled choices and open questions.
3. Make the smallest change that keeps the pipeline runnable end to end.
4. Reusable logic → library; a runnable stage → `pipelines/`; visuals → `app/`.
5. Log any modelling decision with its justification.
6. Run `pytest` and the relevant pipeline stage before declaring done.
