# 01 — Phase 1: Elo baseline (national teams)

End-to-end skeleton: ingest international results → Elo ratings + draw-aware 1X2 → walk-forward RPS backtest → forecast store. Acceptance met: Elo beats the base-rate baseline on RPS.

## Result

Walk-forward, yearly origins 1990–2026, 32,327 test matches, refit each origin:

| model | RPS | log loss | Brier |
|-------|-----|----------|-------|
| **elo** | **0.1809** | 0.9135 | 0.5383 |
| base rate | 0.2257 | 1.0484 | 0.6316 |

Elo improves RPS by **+0.0448** over the base rate → **PASS**. RPS ≈ 0.18 is a credible 1X2 number (bookmaker closing lines sit around 0.18–0.19; we benchmark against de-vigged odds in Phase 4, not yet).

Sanity forecast (model fit asof 2026-03-31): Uruguay vs Cape Verde (home) → H 0.683 / D 0.198 / A 0.119.

## How to reproduce

```bash
# Environment: Python 3.12 .venv (see docs/decisions.md 006)
pip install -e ".[dev,research]"

# 1. Ingest the public, auth-free GitHub mirror of the intl-results dataset.
#    NOTE: the dataset is updated continuously, so absolute counts/metrics drift
#    slightly over time. Pin by snapshotting data/raw/intl_results.csv if exact
#    reproduction is required (raw data is immutable once fetched).
python -m pipelines.ingest
#    → ~49,445 matches, 1872–2026, 327 teams

# 2. Walk-forward backtest (no seed — Elo is deterministic).
python -m pipelines.backtest --model both --start 1990
#    → elo RPS ≈ 0.1809, baseline RPS ≈ 0.2257, acceptance PASS

# 3. Fit on all history and write the store for the app.
python -m pipelines.train    --model elo
python -m pipelines.forecast --model elo --n 200
```

No random seed is involved in Phase 1 (Elo + the 1-D draw-parameter fit are deterministic). Outputs land in `data/processed/`, `artifacts/models/`, `artifacts/metrics/`, `artifacts/forecasts/forecasts.sqlite`.

## Modelling choices (logged in docs/decisions.md)

- **Data source** = martj42 GitHub mirror (auth-free, reproducible) instead of Kaggle.
- **comp_type classifier** = coarse 4-bucket map from tournament name (friendly / qualifier / final_tournament / continental); drives Elo K-factor importance. Refinable.
- **1X2 head** = Davidson tie model on the Elo rating gap, one draw parameter `nu` fit by log loss on pre-match gaps. Keeps standard Elo home/away odds; draw share peaks for even teams.
- **Backtest** = expanding-window, yearly origins, refit per origin, `min_train=500`.

## Known limitations → motivates later phases

- Elo gives **no scoreline distribution** (1X2 only). Phase 2 (Dixon–Coles) fixes this via the keystone.
- The draw model is a single global `nu`; a full ordered logit on comp_type is a refinement.
- comp_type bucketing is coarse; the M01 importance weights are hand-set defaults, not tuned.
- No market benchmark yet (Phase 4).
