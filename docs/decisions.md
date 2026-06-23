# Decisions log

ADR-style running log. **Every modelling and architectural choice ends with a
decision and a one-line justification** (roadmap.md Part 9). Append; don't rewrite
history. Newest at the bottom of each section.

Format:
```
### NNN — <title>   (status: accepted | superseded | open)   <date>
**Decision:** what we chose.
**Why:** one line.
**Notes / supersedes:** optional.
```

---

## Architecture decisions (settled at scaffold time)

### 001 — Heavy compute on the PC, Pi displays only   (accepted) 2026-06-22
**Decision:** All training, MCMC, backtesting run on the PC; the Pi runs only the
read-only dashboard.
**Why:** The arm64 8 GB Pi can't sample PyMC posteriors at usable speed; keeping it
read-only makes it robust and always-on.

### 002 — Batch artifact store as the PC↔Pi handoff   (accepted) 2026-06-22
**Decision:** PC writes forecasts to `artifacts/` (SQLite + parquet); the Pi reads
it. No live Pi→PC call at view time.
**Why:** Decouples the two boxes; dashboard works even when the PC is off; simplest
robust option. On-demand Tailscale compute kept as a future option (see 005).

### 003 — Dashboard is FastAPI + HTMX, read-only   (accepted) 2026-06-22
**Decision:** Server-rendered FastAPI + Jinja2 + HTMX; app imports no modelling
code or heavy deps.
**Why:** Small arm64 image, no JS build step, light enough for the Pi; depends only
on the store schema.

### 004 — Library / pipelines / app separation   (accepted) 2026-06-22
**Decision:** Reusable logic only in `src/football_forecast/`; `pipelines/` are thin
runners; `app/` only reads the store.
**Why:** Keeps the modelling core testable and reusable; keeps the Pi image light.

### 008 — Fixtures store + forecast queue (Pi-writable)   (accepted) 2026-06-22
**Decision:** Upcoming fixtures and user-entered results live in a **separate,
writable** `fixtures.sqlite` (not the read-only, PC-synced forecasts store). Each
row carries its own forecast bundle. The **queue** = rows with no forecast yet;
draining it (`fixtures_queue.drain`) is **cheap inference** with an already-fitted
model, so it runs on the Pi (synced model pickle) or the PC. Adding a *result*
needs no compute here; folding results into model training is a later PC job.
**Why:** Keeps the one-way PC→Pi sync intact (no two-writer clobber of the
forecasts store), while letting the Pi accept fixtures/results and compute forecasts
without training — consistent with "Pi displays + cheap inference, never trains"
(001). The DAL (`store/fixtures.py`) and drainer are stdlib/light so the app uses
them without pulling pandas/modelling deps.
**Seeding:** `pipelines.wc2026` registers the real WC2026 group matches from the
results data with **pre-tournament** forecasts (model fit asof the tournament
start — no leakage).

### 009 — MCMC sampler = nutpie (numba), no C compiler   (accepted) 2026-06-23
**Decision:** Sample the Bayesian model with **nutpie** (numba backend), not
PyMC's default pytensor-C sampler.
**Why:** No g++/clang on the box; the pytensor fallback is unusably slow. nutpie
JITs via numba and needs no C toolchain — a 50-team/2000-match hierarchical model
samples in ~8s. Supersedes the "install a C compiler before Phase 3" note (006).
Adds `nutpie`/`numba` to the `research` extra.

### 010 — League data source = GitHub mirror of football-data.co.uk   (accepted) 2026-06-23
**Decision:** Use the consolidated GitHub mirror
(`xgabora/Club-Football-Match-Data-2000-2025`, one `Matches.csv`) for league data.
**Why:** football-data.co.uk is unreachable from this environment (connection
times out; only GitHub egress works). The mirror carries goals/shots/fouls/corners/
cards + 1X2 & O/U odds for 42 leagues 2000–2025 — everything except **referee**
(so M06 stays deferred). Loader: `data/sources/club_matches.py`; one division per
processed parquet.

### 005 — On-demand PC compute over Tailscale   (open) 2026-06-22
**Decision:** Deferred. Revisit only if a live "recompute with the Bayesian model"
feature is needed in the dashboard.
**Why:** The batch store covers all current needs; adding a networked service isn't
justified yet.

### 006 — Python 3.12 in a local .venv   (accepted) 2026-06-22
**Decision:** Develop on Python 3.12 (`E:\Program Files\Python12`), venv at `.venv/`.
**Why:** The Bayesian/ML stack (PyMC/pytensor, LightGBM, arviz) is unreliable on the
freshly-released 3.14; 3.12 is the stable sweet spot. Deps split into a light core
(also installed on the Pi) and a heavy `research` extra (PC only).
**Note (perf, not blocking):** the full `research` stack installs and imports
cleanly (PyMC 6.0.1, pytensor 3.0.7, LightGBM 4.6.0, arviz 1.2.0). pytensor warns
`g++ not available` → PyMC's default sampler runs in a slow fallback.
**Resolved for Phase 3 without a compiler:** we sample with **nutpie** (numba
backend), which JITs the model and needs no C toolchain — a 50-team/2000-match
hierarchical model samples in ~8s. So the "install a C compiler" step is no longer
required (superseded by decision 009).

### 007 — Competition-agnostic core, World Cup as first target   (accepted) 2026-06-22
**Decision:** Near-term validation focus is the **World Cup**, but the data model,
store schema, and modelling code stay **competition-agnostic** (a `competition`
field everywhere) so the same pipeline extends to leagues, continental, and club
competitions without redesign.
**Why:** Gives a concrete first milestone while preserving the long-term goal of
forecasting any football competition. Avoids baking World-Cup-specific assumptions
into the core.

---

## Modelling decisions (open — resolve as phases proceed)

These come from roadmap.md Part 9. Each must end with a decision + justification.

### M01 — National-team match scope   (accepted) 2026-06-22
**Decision:** Include **all internationals** (friendlies + qualifiers + finals);
encode **competition type** as a feature rather than filtering matches out.
**Why:** National teams play few matches/year — dropping friendlies throws away
scarce data. Keeping them with an intensity/type feature lets the model learn the
difference instead of us assuming it. Intensity weighting can be tuned later (M02).

### M02 — Time-decay half-life for match weighting   (accepted) 2026-06-22
**Decision:** Default Dixon–Coles half-life = **1460 days (~4 years)**.
**Why:** Backtest sweep (origins 2005–2026, `research/exp_halflife.py`): no decay
0.1814 RPS, 365d 0.1753, 730d 0.1727, **1460d 0.1723** (best); gains flatten past
~730d. No-decay is dramatically worse → recency weighting is the key ingredient.
4 years suits international football (squads turn over slowly over a WC cycle).

### M03 — Home advantage for neutral-venue matches   (accepted) 2026-06-22
**Decision:** On neutral-venue matches, **drop the home-advantage term** (Elo: no
home bonus; Poisson: home-advantage applies only when `neutral == false`). The
results dataset's neutral-venue flag drives this.
**Why:** There is no host on a neutral pitch; applying a home bonus would bias
tournament matches (exactly the World Cup case we care about first).

### M04 — Poisson vs negative binomial per target   (accepted) 2026-06-23
**Decision:** Auto-select per target on residual var/mean (> 1.15 → NegBin). On E0:
**corners → NegBin** (1.43); **goals, fouls, yellow cards → Poisson** (~0.9–1.15).
**Why:** Decided empirically (Phase 4). Notably fouls/yellows are *not*
overdispersed here — the roadmap's prior guess didn't hold; the data decides.

### M05 — Training-window length (volume vs relevance)   (deferred → tune in backtest)
**Trigger:** Phase 1–2 backtesting; tune alongside M02 (decay vs hard window).

### M06 — Referee as fixed vs random effect (cards/fouls)   (deferred — no data) 2026-06-23
**Status:** Still deferred. The reachable league source (GitHub mirror) **drops the
referee column**, and football-data.co.uk (which has it) is unreachable here. Count
models currently use team for/against + home, no referee term. Revisit when a
referee-bearing source is reachable.

### M07 — Catalogue of leakage fields per chosen schema   (accepted) 2026-06-23
**Decision:** Documented per source in `docs/data.md` → "Leakage catalogue". For the
league schema: all match-outcome stats (goals, shots, fouls, corners, cards, FT/HT
results) are **targets, never features** for the same match; pre-kickoff fields
(odds, Elo snapshots, rolling form) are safe. Odds are used only as the benchmark,
never as a model input.

---

## Phase 1 implementation decisions   (accepted) 2026-06-22

### P1-01 — National-team data source = martj42 GitHub mirror
**Decision:** Ingest the international results dataset from its public GitHub raw
CSV (`martj42/international_results`), not Kaggle.
**Why:** Auth-free → reproducible without a Kaggle account/API key. Same data as
the Kaggle dataset. Trade-off: it updates continuously, so pin
`data/raw/intl_results.csv` for exact reproduction.

### P1-02 — Coarse comp_type classifier
**Decision:** Map tournament name → {friendly, qualifier, final_tournament,
continental} with a 4-rule classifier; this drives Elo K-factor importance.
**Why:** Enough signal for the M01 importance feature in Phase 1; a fuller
competition taxonomy is a refinement, not a blocker.

### P1-03 — Elo 1X2 head = Davidson tie model, single `nu`
**Decision:** Turn the Elo rating gap into H/D/A with a Davidson tie model whose
one draw parameter `nu` is fit by log loss on pre-match gaps during `fit`.
**Why:** Keeps standard Elo home/away odds exactly, adds a principled draw
probability that peaks for even teams, learns the draw rate from data, no leakage.
Refinement to a full ordered logit on comp_type is logged for later.

### P1-04 — Backtest protocol = expanding window, yearly origins, refit per origin
**Decision:** Walk-forward with Jan-1 origins, refit each origin, `min_train=500`.
**Why:** Strictly time-ordered, no leakage; yearly granularity keeps refits cheap
at 49k matches while giving 30+ folds.

**Phase 1 result:** Elo RPS 0.1809 vs base-rate 0.2257 over 32,327 test matches
(1990–2026) → acceptance PASS. Full reproduce recipe in
`research/notes/01-phase1-elo.md`.

---

## Phase 2 implementation decisions   (accepted) 2026-06-22

### P2-01 — Keystone implemented once in forecast/
**Decision:** `scoreline_matrix()` + `markets.py` (one_x_two/over_under/btts/
correct_score) are the single source for all markets; truncate at max_goals=10
and renormalize.
**Why:** 1X2/OU/CS must never be modelled independently — all are sums over the
one matrix (docs/models-explained.md §3). Golden-number tested.

### P2-02 — Goal-model fitting = penalized MLE with analytic gradient
**Decision:** Fit attack/defence/home/intercept by L-BFGS-B on the weighted
Poisson NLL with an analytic O(N) gradient and an L2 penalty on attack/defence.
**Why:** Analytic gradient makes the 2T+2 params tractable per refit; the L2
penalty removes the additive degeneracy AND shrinks sparse teams toward average
(the shrinkage the roadmap calls for).

### P2-03 — Dixon–Coles rho via two-stage profile fit
**Decision:** Fit the low-score `rho` in a second 1-D stage, holding the Poisson
means fixed.
**Why:** Standard, fast, stable approximation to the joint DC fit; rho is a small
refinement on top of the means, so profiling it loses little and avoids a harder
joint optimization.

### P2-04 — Maher kept as an ablation
**Decision:** `MaherModel` = independent Poisson, no decay, no rho — retained in
the comparison.
**Why:** It isolates *what* helps. Finding: Maher (0.1844 RPS) is *worse* than Elo
(0.1809); Dixon–Coles (decay + rho) is what beats Elo. Honest, informative.

**Phase 2 result:** see `research/notes/02-phase2-dixon-coles.md` (Dixon–Coles
beats Elo on RPS and log loss; acceptance PASS).

---

## Phase 3 implementation decisions   (accepted) 2026-06-23

### P3-01 — Single-holdout eval for the Bayesian model (not full walk-forward)
**Decision:** Evaluate the Bayesian model on one recent time-ordered holdout
(`research/exp_phase3.py`), not the 36-origin walk-forward used for cheap models.
**Why:** MCMC per fit is ~10s+; a full walk-forward is impractical. One honest
holdout suffices to compare against Dixon–Coles on identical splits.

### P3-02 — Recency via hard window, time-decay off
**Decision:** Bayesian model trains on a 2000-day window; the exponential
time-decay knob (`half_life_days`) defaults off.
**Why:** Empirically the window alone beat window+decay (0.1639 vs 0.1657 RPS) —
the two recency mechanisms double-count. Knob retained for experiments.

### P3-03 — Convergence gating via self-contained split-R-hat
**Decision:** Compute max split-R-hat in-house (not via arviz) and expose a
`converged` flag; `strict=True` refuses to emit forecasts from a non-converged fit.
**Why:** Robust to arviz API differences; a bad fit must never reach the store.

**Phase 3 result (qualified PASS):** converges (max R-hat 1.009); partial pooling
demonstrated emphatically (corr(log #matches, attack sd) = −0.972); **slightly
trails** the well-tuned Dixon–Coles on aggregate holdout RPS (0.1639 vs 0.1593) —
the honest outcome roadmap 3.5 anticipates. The Bayesian model's value is
calibrated uncertainty for sparse teams, not aggregate point accuracy. Full
write-up: `research/notes/03-phase3-bayesian.md`.

---

## Phase 4 implementation decisions   (accepted) 2026-06-23

### P4-01 — Count models = penalized Poisson MLE mean + family by dispersion
**Decision:** `models/counts.py` fits log E[count] = mu + home + for_[team] +
against_[opp] by penalized Poisson MLE (same machinery as the goal model); the
predictive family (Poisson/NegBin) is auto-chosen from residual dispersion (M04).
**Why:** Reuses proven, fast machinery; lets each target pick its own family from
the data.

### P4-02 — Market benchmark = proportional de-vig of closing odds
**Decision:** `eval/market.py` de-vigs 1X2 odds by inverse-odds normalization;
treated as the gold-standard baseline, never as a model input (leakage, M07).
**Why:** Standard, simple baseline; Shin/power methods are a logged refinement.

### P4-03 — One processed parquet per division
**Decision:** `pipelines.ingest_league --division E0` writes
`data/processed/league_<div>.parquet`; models/experiments take a `--division`.
**Why:** Keeps leagues isolated and competition-agnostic (decision 007).

**Phase 4 result (PASS, M06 deferred):** On E0 (7,600 matches), Dixon–Coles 1X2
RPS 0.2047 vs de-vigged market 0.1948 (base rate 0.2313) — within 0.01 of the
market. Count models calibrated (coverage near nominal). M04 resolved (corners →
NegBin; others Poisson). Referee (M06) unavailable in the source. Full write-up:
`research/notes/04-phase4-leagues.md`.

---

## Phase 5 implementation decisions   (accepted) 2026-06-23

### P5-01 — Leakage-free features via a single chronological pass
**Decision:** `features/engineering.py` maintains per-team state (Elo, rolling
form, rest days), snapshots it **before** each match, then updates. The post-train
state predicts future fixtures.
**Why:** Guarantees no leakage and matches how Elo/DC behave in the backtester
(fit once at origin, static within fold).

### P5-02 — Boosting behind the same protocol; 1X2 multiclass, counts Poisson
**Decision:** `BoostingModel` (LightGBM multiclass) implements OutcomeModel and
drops into the walk-forward harness; `BoostingCountModel` uses the Poisson
objective for count targets. Added to the backtest/train/forecast/queue factories.
**Why:** Apples-to-apples comparison on identical splits.

**Phase 5 result (PASS — honest negative finding):** On E0, gradient boosting
**loses on 1X2** (RPS 0.2269 vs Elo 0.2039 / DC 0.2042, and poorly calibrated
ECE 0.091) and **ties on counts** (corners: boosting MAE 2.238 vs GLM 2.253; GLM
deviance 1.577 vs 1.580). The structured goal models' inductive bias wins on 1X2;
boosting only matches where there's no generative structure to exploit (counts).
The de-vigged market (RPS 0.1938) remains the ceiling. Full synthesis:
`research/notes/05-phase5-boosting.md`.

## Project synthesis (all phases)

Dixon–Coles is the workhorse (best/tied-best 1X2 on both national teams and
leagues, cheap, calibrated). Bayesian adds calibrated uncertainty for sparse teams
but doesn't beat DC on aggregate. Boosting honestly loses on 1X2, ties on counts.
The market is the gold standard none beat (DC within ~0.01 RPS). Recurring lesson:
**inductive bias + recency weighting beat model complexity** on this data.
