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

### M02 — Time-decay half-life for match weighting   (deferred → tune in backtest)
**Trigger:** Phase 1–2 backtesting. **Plan:** treat the half-life as a hyperparameter,
select by backtest RPS on time-ordered splits. Until then, start with a moderate
default and record it with each run.

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
