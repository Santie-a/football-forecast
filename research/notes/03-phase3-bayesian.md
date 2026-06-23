# 03 — Phase 3: Bayesian hierarchical model (national teams)

Baio–Blangiardo hierarchical Poisson in PyMC, sampled with **nutpie** (numba
backend — no C compiler needed). Partial pooling shrinks sparse teams toward the
population mean and yields **posterior-predictive** scoreline forecasts that carry
parameter uncertainty.

## Result (single recent holdout)

MCMC is expensive, so Phase 3 uses a single time-ordered holdout rather than the
full walk-forward. Train on data before 2025-06-01 (Bayesian: 2000-day window),
test on [2025-06-01, 2026-06-22) — **1139 matches**, identical split for both.

| model | RPS | log loss | converged |
|-------|-----|----------|-----------|
| dixon_coles | **0.1593** | **0.8317** | — |
| bayesian (hierarchical) | 0.1639 | 0.8486 | ✅ max R-hat 1.009 |

### Acceptance — qualified PASS

- **Convergence: PASS.** max R-hat 1.009 < 1.01 (tune 2500, 4 chains). A
  non-converged fit is gated (`converged` property; `strict=True` refuses to emit).
- **Wider intervals for low-data teams: emphatic PASS.** corr(log #matches,
  posterior attack sd) = **−0.972**. Chagos Islands (1 match) → attack sd 0.549;
  Mexico (83 matches) → 0.103. This is the whole point of partial pooling, and the
  model delivers it cleanly — something the MLE models (Elo, Dixon–Coles) cannot
  express at all.
- **Beats/matches Dixon–Coles: trails slightly.** −0.0047 RPS / −0.0168 log loss.
  Comparable, not better.

### The honest finding

On aggregate 1X2 over a holdout dominated by data-rich teams, the well-tuned
Dixon–Coles edges the Bayesian hierarchical model. This is exactly the outcome
roadmap Part 3.5 anticipates ("be ready to find — and report honestly — that a
well-tuned Dixon–Coles often beats more complex models"). The Bayesian model's
value is **not** a better aggregate point forecast; it is **calibrated uncertainty**
— honest, wide distributions for the many sparse national teams, and a full
posterior we can interrogate. For a World-Cup minnow with a handful of matches,
that humility is the right behaviour even if it barely moves aggregate RPS.

Two recency mechanisms were tried: a hard training **window** (2000 d) and an
exponential **time-decay** on the likelihood. Window-only won; adding decay on top
slightly hurt (0.1657 vs 0.1639) — they double-count recency. Decay left **off** by
default (the `half_life_days` knob remains for experiments).

## How to reproduce

```bash
pip install -e ".[dev,research]"        # includes nutpie + numba (no C compiler needed)
python -m research.exp_phase3 --origin 2025-06-01 --end 2026-06-22
#   → dixon_coles RPS 0.1593 | bayesian RPS 0.1639, R-hat 1.009 (converged)
#   → partial pooling corr -0.972
```

Seeded (`seed=1`); nutpie sampling is deterministic given the seed, so numbers
reproduce up to minor platform differences. Pin `data/raw/intl_results.csv` for
exact data (the source updates continuously).

## Modelling choices (logged in docs/decisions.md, 009 + P3-xx)

- **Sampler = nutpie / numba** — fast NUTS with no C toolchain (resolves the
  decision-006 compiler note).
- **Recency = hard window (2000 d), decay off** — empirically better than decay.
- **Convergence gating** via a self-contained split-R-hat (no arviz API churn);
  non-converged fits are refused under `strict`.
- **Posterior-predictive forecasts** — scoreline matrix averaged over draws; a
  model-level `scoreline()` that `markets_for_model` prefers over point `rates()`.

## Known limitations → next phases

- Single holdout, not full walk-forward (MCMC cost). A multi-origin Bayesian
  backtest is feasible but slow; deferred.
- Independent Poisson per draw (no Dixon–Coles `rho` inside the Bayesian model) —
  could add a low-score correction to the posterior-predictive matrix.
- Unknown teams fall back to the population mean (atk=def=0) without extra
  predictive spread; a fuller treatment would sample new-team effects from the prior.
- The real payoff (calibration on sparse teams) deserves a reliability-diagram
  comparison vs DC — a good follow-up.
