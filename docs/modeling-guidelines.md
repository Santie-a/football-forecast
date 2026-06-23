# Modelling guidelines

Operational rules distilled from roadmap.md Parts 3–6. These are conventions the
code must follow, not the research itself (that stays in roadmap.md + research notes).

## Model families (build in this order)

Each is a baseline for the next; keep them all so you can compare honestly.

1. **Elo** (+ football/national-team variants) → convert rating diff to H/D/A.
   Also: ordered logit/probit as a strength-diff baseline.
2. **Maher / Dixon–Coles** Poisson goal models → the core. Produces λ_home, λ_away.
3. **Extensions:** bivariate Poisson (correlation), negative binomial (overdispersion).
4. **Bayesian hierarchical** (Baio–Blangiardo, PyMC) → partial pooling for sparse
   national teams; full posterior-predictive scorelines. **PC-only (MCMC).**
5. **Gradient boosting** (LightGBM/XGBoost) with **Poisson/Tweedie objectives** for
   counts, multiclass for 1X2 — for comparison. Expect Dixon–Coles/Bayesian to
   often win 1X2; report that honestly if so.
6. **Separate count models** (Poisson/NegBin) for fouls/cards/corners, conditioned
   on **referee** + team style + game state. Feasible mainly at league level.

## The keystone (implement once, reuse everywhere)

A goal model gives λ_home, λ_away. From those:

1. Build the **scoreline matrix** P(home=i, away=j) — product of two Poisson PMFs
   (under independence), with the Dixon–Coles low-score correction where used.
2. Derive **everything else by summing the matrix**: 1X2 (i>j / i=j / i<j),
   over/under (sum where i+j ≷ line), correct-score (individual cells).

Lives in `src/football_forecast/forecast/`. 1X2 / over-under / correct-score must
**never** be modelled independently — they come from the matrix.

For counts: the count model's predicted mean *is* the expected count; its
distribution gives the over/under lines.

## Backtesting — non-negotiable rules

- **Strictly time-ordered, walk-forward** splits. Train only on the past.
- **Never shuffle matches into random train/test splits.** This leakage is silent
  and makes bad models look excellent. It is the #1 bug risk in this project.
- Re-fit (or update) the model at each split boundary as new matches arrive.
- The backtester is shared infra in `src/football_forecast/eval/` — every model
  is evaluated through the same harness on identical splits.

## Evaluation — proper scoring only

- **Never use accuracy** for probabilistic forecasts.
- **RPS (Ranked Probability Score)** — primary metric for 1X2 (it's ordinal).
- **Log loss** and **Brier score** — secondary.
- **Calibration** — reliability diagrams; do "20%" events happen ~20% of the time?
- **Count targets** — Poisson deviance + posterior-predictive coverage checks.
- **Market benchmark** — convert bookmaker odds to **de-vigged implied
  probabilities**; treat the closing line as the gold-standard baseline. Getting
  *close* is already a strong, credible result.

Standard comparative design: **Elo/ordered-logit → Dixon–Coles → Bayesian
hierarchical → gradient boosting**, all on the same time splits with RPS + log
loss + calibration.

## Reproducibility

- Fix random seeds; record them with each run.
- A model run writes: the fitted artifact, the config/seed used, and its backtest
  metrics — so any forecast in the store is traceable to how it was produced.
- Log every modelling choice (and its one-line justification) in
  [`decisions.md`](decisions.md). roadmap.md Part 9 lists the open questions to resolve.
