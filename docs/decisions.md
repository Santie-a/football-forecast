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
`g++ not available` → PyMC runs in a slower fallback. Irrelevant for Phases 1–2;
**before Phase 3 (MCMC)**, install a C toolchain (e.g. m2w64 `gxx`/mingw-w64) on
the PC for fast sampling.

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

### M04 — Poisson vs negative binomial per target   (deferred → decide from EDA)
**Trigger:** EDA dispersion check (variance vs mean) per target. **Expectation:**
goals ≈ Poisson; fouls/cards likely overdispersed → NegBin. Decide empirically.

### M05 — Training-window length (volume vs relevance)   (deferred → tune in backtest)
**Trigger:** Phase 1–2 backtesting; tune alongside M02 (decay vs hard window).

### M06 — Referee as fixed vs random effect (cards/fouls)   (deferred → Phase 4)
**Trigger:** League phase, where referee data and count targets exist. Not relevant
to the national-team / World Cup phases.

### M07 — Catalogue of leakage fields per chosen schema   (open — ongoing)
**Plan:** Populate as each source schema is examined; record every post-kickoff
field that must never be a feature. See `docs/data.md`.

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
