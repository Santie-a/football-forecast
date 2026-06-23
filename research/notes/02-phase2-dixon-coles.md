# 02 — Phase 2: Dixon–Coles (national teams)

The keystone (scoreline matrix → markets) plus Poisson goal models: Maher (independent) and Dixon–Coles (low-score correction + time-decay). Acceptance met: Dixon–Coles beats Elo on **both** RPS and log loss.

## Result

Walk-forward, yearly origins 1990–2026, 32,327 test matches, refit each origin:

| model | RPS | log loss | Brier |
|-------|-----|----------|-------|
| base rate | 0.2257 | 1.0484 | 0.6316 |
| Elo (Phase 1) | 0.1809 | 0.9135 | 0.5383 |
| Maher (indep. Poisson) | 0.1844 | 0.9268 | 0.5447 |
| **Dixon–Coles** | **0.1738** | **0.8874** | **0.5206** |

Dixon–Coles beats Elo by **+0.0071 RPS** and **+0.0261 log loss** → **PASS**.

### The interesting finding

**Maher alone is *worse* than Elo** (0.1844 vs 0.1809). The attack/defence/home structure by itself doesn't beat a well-tuned Elo. What pushes Dixon–Coles ahead is the **time-decay weighting** (and, secondarily, the low-score `rho`): the half-life sweep shows no-decay Maher-style fitting at 0.1814 RPS vs 0.1723 at a 4-year half-life. Lesson: on this data, *recency weighting matters more than model structure*.

### Half-life selection (resolves M02)

Sweep (`research/exp_halflife.py`, origins 2005–2026, Dixon–Coles):

| half-life (days) | RPS |
|------------------|-----|
| none (equal weight) | 0.1814 |
| 365 | 0.1753 |
| 730 | 0.1727 |
| **1460** | **0.1723** |

Chosen default **1460 days (~4 years)** — best RPS; the curve flattens past ~730d. Logged as decision M02.

## How to reproduce

```bash
pip install -e ".[dev,research]"
python -m pipelines.ingest                          # ~49,445 matches (pin data/raw for exact numbers)
python -m pipelines.backtest --model all --start 1990
#   → dixon_coles RPS ≈ 0.1738, log loss ≈ 0.8874; Phase 2 PASS
python -m research.exp_halflife --start 2005        # the M02 half-life sweep
python -m pipelines.forecast --model dixon_coles    # writes DC 1X2 to the store
```

No random seed: the goal-model fit (L-BFGS-B) is deterministic given the data; the only stochastic element is the synthetic data in the unit tests (seeded).

## Modelling choices (logged in docs/decisions.md, P2-01…P2-04, M02)

- **Keystone** implemented once in `forecast/`; all markets are sums over the matrix.
- **Fitting** = penalized MLE (analytic O(N) gradient, L-BFGS-B) with L2 shrinkage on attack/defence (removes degeneracy + helps sparse teams).
- **rho** fit in a two-stage profile (means fixed) — fast, standard approximation.
- **Maher** retained as an ablation to isolate what helps.

## Known limitations → next phases

- Point estimates only — no parameter uncertainty. Phase 3 (Bayesian hierarchical) adds posterior-predictive forecasts and should help sparse teams further.
- `rho` two-stage (not joint) — a small approximation; revisit if it matters.
- Still no scoreline distribution surfaced in the dashboard or benchmarked against the market (Phase 4).
- comp_type importance weights and the L2 `reg` strength are hand-set, not tuned.
