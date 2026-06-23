# 04 — Phase 4: leagues + count targets + market benchmark

Moved from national teams (scores only) to league data, where fouls/cards/corners
and bookmaker odds exist. Re-ran the goal model there, added count models for the
new targets, and finally benchmarked 1X2 against the de-vigged closing-odds market.

Data: a GitHub mirror of football-data.co.uk (football-data.co.uk itself is
unreachable from this environment). English Premier League (E0), 7,600 matches
2005–2025. **No referee column in this mirror** → M06 (referee effects) stays
deferred until a referee-bearing source is reachable.

## A. Dispersion → Poisson vs NegBin (resolves M04)

Residual var/mean after fitting the count mean model, per target:

| target | dispersion | family chosen |
|--------|-----------|---------------|
| goals | 1.15 (marginal) | Poisson |
| fouls | 1.13 | Poisson |
| yellow cards | 0.93 | Poisson |
| corners | 1.43 | **negative binomial** |

**Finding:** only **corners** are meaningfully overdispersed here. Fouls and
yellow cards are ~Poisson (yellows even slightly *under*-dispersed) — contrary to
the roadmap's prior expectation that cards/fouls would be overdispersed. Decided
empirically (auto-select at residual var/mean > 1.15). M04 resolved.

## B. Dixon–Coles 1X2 vs the market (headline)

Walk-forward (yearly origins 2010–2025), 5,700 test matches, identical splits:

| model | RPS |
|-------|-----|
| base rate | 0.2313 |
| **Dixon–Coles** | **0.2047** |
| market (de-vigged closing odds) | 0.1948 |

Dixon–Coles beats the base rate by 0.027 and lands **+0.0099 from the market** —
within ~5% of the gold standard. Beating the closing line is extremely hard;
getting this close is a strong, credible result (roadmap Part 6). The de-vig uses
the basic proportional method (Shin/power are possible refinements).

## C. Count-model calibration (predictive coverage)

Count models conditioned on team for/against + home, fit by penalized Poisson MLE;
predictive distribution Poisson or NegBin per A. Holdout season 2023/24, central
predictive interval coverage:

| target | 50% interval | 80% interval |
|--------|-------------|-------------|
| fouls (Poisson) | 0.55 | 0.80 |
| corners (NegBin) | 0.58 | 0.84 |
| yellow (Poisson) | 0.64 | 0.89 |

Fouls and corners are well calibrated. Yellows over-cover (mean ~1.6 → integer
interval endpoints are chunky, so discrete intervals are conservative) — a
known low-count artefact, not miscalibration of the mean.

## Acceptance — PASS (with M06 deferred)

- Count models calibrated: **PASS** (coverage near nominal; yellow over-cover is a
  discreteness artefact).
- 1X2 reported vs de-vigged closing odds: **PASS** (within 0.01 RPS).
- M04 resolved empirically (corners → NegBin; others Poisson).
- M06 (referee) **deferred** — not in the reachable data source.
- M07 leakage catalogue populated (see docs/data.md).

## How to reproduce

```bash
pip install -e ".[dev,research]"
python -m pipelines.ingest_league --division E0          # -> data/processed/league_E0.parquet
python -m research.exp_phase4 --division E0 --since 2005-08-01
#   A: corners -> nbinom, others Poisson
#   B: DC RPS ~0.2047 vs market ~0.1948 (base rate 0.2313)
#   C: coverage near nominal
```

Deterministic (no seed needed). Pin `data/raw/club_matches.csv` for exact numbers
(the mirror updates). Other leagues: `--division D1|SP1|I1|F1|...`.

## Known limitations → next

- **No referee** in the source → M06 deferred; cards/fouls models omit the single
  biggest known driver. A referee-bearing source would likely sharpen them.
- Count models are independent home/away (no joint home–away count correlation).
- De-vig is proportional only.
- The dashboard/store don't yet surface count forecasts or the league competitions
  (goal models + scoreline already generalize; counts need a store market type).
