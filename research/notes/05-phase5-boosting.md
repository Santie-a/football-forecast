# 05 — Phase 5: gradient boosting + synthesis

The ML challenger (LightGBM) on engineered pre-match features, and the honest
cross-model synthesis the project has been building toward.

## Setup

- Features (`features/engineering.py`, leakage-free single pass): Elo diff, rolling
  form (points/goals for & against, last 10), rest days, neutral flag.
- `models/boosting.py`: `BoostingModel` (multiclass 1X2) + `BoostingCountModel`
  (Poisson regressor for counts). Both behind the usual protocol.
- Evaluated on E0, walk-forward yearly origins, identical splits, vs the de-vigged
  closing-odds market.

## A. 1X2 comparison (4,940 test matches)

| model | RPS | log loss | ECE (calibration) |
|-------|-----|----------|-------------------|
| market (de-vigged) | **0.1938** | 0.954 | 0.008 |
| Elo | 0.2039 | 0.986 | 0.007 |
| Dixon–Coles | 0.2042 | 0.986 | 0.006 |
| gradient boosting | 0.2269 | 1.098 | 0.091 |
| base rate | 0.2325 | 1.067 | 0.012 |

## B. Count target (corners): structured GLM vs boosting

| model | MAE | Poisson deviance |
|-------|-----|------------------|
| GLM count model | 2.253 | **1.577** |
| LightGBM (Poisson) | **2.238** | 1.580 |

## The synthesis — which model wins which target, and why

**1X2 → the structured goal models win decisively; boosting loses.** Elo and
Dixon–Coles (~0.204 RPS, ECE ~0.006) sit just behind the market and are beautifully
calibrated. Gradient boosting (0.2269, ECE 0.091) is **worse than Elo and barely
beats the base rate**, and is poorly calibrated. This is exactly roadmap Part 3.5's
prediction. *Why:* a goal model bakes in the right inductive bias — goals are
Poisson, a scoreline matrix yields coherent, normalized 1X2 — so it needs little
data and can't produce silly probabilities. Boosting must *learn* that structure
from ~13 noisy features and a few thousand rows; it overfits and miscalibrates.
The closing line stays the ceiling: it aggregates far more information than any
single model.

**Counts → boosting and the GLM tie.** Corners (the one overdispersed target) have
no elegant generative structure to exploit, so the problem is closer to plain
regression where boosting's flexibility matches the GLM: boosting wins MAE by a
hair (2.238 vs 2.253), the GLM wins deviance by a hair. Honest verdict: a wash —
use whichever is simpler (the GLM, which also gives a calibrated predictive
distribution for over/under).

**Across the whole project:**
- **Dixon–Coles is the workhorse** — best or tied-best on 1X2 for both national
  teams (Phase 2: 0.1738 RPS, beat Elo) and leagues (Phase 4/5: ~0.204, within 0.01
  of the market), cheap and well-calibrated.
- **Bayesian hierarchical** (Phase 3) doesn't beat DC on aggregate but adds what DC
  can't: calibrated uncertainty for sparse teams (the only model that's *humble*
  about a team it has barely seen).
- **Boosting** is a legitimate challenger that honestly loses on 1X2 and ties on
  counts — a result worth reporting precisely because it's negative.
- **The market** is the gold standard none of them beat — closeness (DC within
  ~0.01 RPS) is the real achievement.
- The recurring lesson across phases: **inductive bias + recency weighting beat
  model complexity** on this data.

## How to reproduce

```bash
pip install -e ".[dev,research]"
python -m pipelines.ingest_league --division E0
python -m research.exp_phase5 --division E0 --since 2005-08-01
#   A: market 0.1938 < elo 0.2039 ≈ dc 0.2042 < boosting 0.2269 < base 0.2325
#   B: corners GLM ≈ boosting
```

LightGBM is seeded (`random_state`); numbers reproduce up to minor platform
differences. Pin `data/raw/club_matches.csv` for exact data.

## Limitations / further work

- A richer feature set (rolling xG, lineups, market-implied strength) might lift
  boosting — but the structured models would likely still lead on 1X2.
- Stacking/ensembling DC + boosting + market is the natural next step if chasing
  the closing line.
- xG features need a source with shot data (FBref/Understat via `soccerdata`).
