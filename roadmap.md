# Football Match Forecasting — Research Roadmap

> A self-study and build plan. The goal is not just a model that outputs numbers, but
> one whose internals you understand from first principles. Each section lists **what to
> research** (as questions/topics to resolve yourself) rather than finished answers.

**Targets to forecast (per match):**

1. Full **scoreline distribution** — P(home = i, away = j) for every plausible (i, j), e.g. 2–0: 20%, 1–1: 10%.
2. **1X2** — P(home win), P(draw), P(away win).
3. **Expected counts** — fouls, cards (yellow/red), corners, for each team.

**Path:** national-team matches first, then club/league matches.

---

## How to use this document

- Work top to bottom. Earlier parts are prerequisites for later ones.
- For each research topic, write your own short summary (½–1 page) before moving on. If you
  can't explain it without notes, you don't understand it yet.
- Keep a running "decisions log" (see Part 9) — every modelling choice should be justified.
- Treat the **keystone** (Part 4) as the conceptual center: almost everything you forecast
  is derived from a goal-generating model, not predicted directly.

---

## Part 0 — Scope and framing decisions (do this first)

Before any code or math, resolve what you are actually modelling. Research questions:

- **What is the difference between predicting goals and "expected goals" (xG)?** Confirm
  you understand that xG is a *shot-quality metric* (an input/feature), while you are
  forecasting *goals to be scored* (an output). Write the distinction in one paragraph.
- **What does "expected fouls/cards/corners" mean precisely?** Is it the mean of a predicted
  count distribution, the full distribution, or both? Decide and write it down.
- **National teams vs. clubs — why does it change the project?** Research and note:
  - Small samples per team, few matches/year, long gaps, changing rosters.
  - Friendlies vs. competitive matches (qualifiers, finals) — different intensity/effort.
  - **Data availability asymmetry:** scorelines exist for almost all internationals back
    ~150 years, but per-match *fouls/cards/corners* are scarce for national teams and only
    consistently available for major tournaments. They are abundant at league level.
- **Consequence for sequencing:** plan to build **scoreline + 1X2 for national teams first**,
  and add the **fouls/cards/corners count models when you move to leagues**, where the data
  to fit and validate them actually exists. State this constraint explicitly in your write-up.

---

## Part 1 — Mathematical and statistical foundations to (re)establish

You have the background; this is about mapping it onto the problem. Research/refresh:

### 1.1 Count-data distributions
- **Poisson distribution**: assumptions, PMF, why it is the natural model for goals.
- **Overdispersion**: what it is, how to detect it (variance > mean), why goals are roughly
  Poisson but **cards and fouls usually are not**.
- **Negative binomial** as the overdispersed alternative; relationship to Poisson.
- **Zero-inflation** — relevant for red cards (mostly zero).

### 1.2 Joint and correlated counts
- **Independence assumption** for home/away goals — when it fails (low scores, "killing the game").
- **Bivariate Poisson** — how it introduces correlation; what the correlation parameter means.

### 1.3 Estimation
- **Maximum likelihood estimation** for these models: write the likelihood for a season of
  matches under a Poisson goal model; understand what is being optimized.
- **Regularization / shrinkage** and why it matters for sparse national-team data.

### 1.4 Bayesian inference (high priority for this project)
- Prior, likelihood, posterior; **posterior predictive distribution** (this is what gives you
  scoreline probabilities directly).
- **Hierarchical models / partial pooling** — the key idea that lets weak teams "borrow"
  information; ideal for small national-team samples.
- **MCMC** at a working level (what it does, how to read trace plots / convergence
  diagnostics like R-hat), via PyMC or Stan.

### 1.5 Probabilistic forecasting concepts
- **Calibration vs. sharpness** — what a "good" probabilistic forecast means.
- **Proper scoring rules** (preview of Part 6): why accuracy is the wrong metric.

---

## Part 2 — Data: acquisition and understanding

### 2.1 Sources to evaluate (research what each does and does not contain)
- **National-team results (scores only):** the long-history international results dataset
  (e.g. Kaggle "International football results from 1872 to present"). Contains results,
  tournament, neutral-venue flag — **no fouls/cards/corners**.
- **National-team tournaments with detailed events:** StatsBomb open data
  (`github.com/statsbomb/open-data`, Python via `statsbombpy`). Free, event-level (passes,
  shots, fouls, cards with coordinates) for selected tournaments (World Cups, Copa América,
  AFCON, Women's WC). Lets you *derive* fouls/cards/corners and your own xG.
- **League matches with the exact stats you want:** football-data.co.uk. Free CSVs from
  2000/01 to present with goals, shots, corners, **fouls, yellow/red cards, referee**, plus
  bookmaker odds. This is your main source once you move to leagues.
- **Richer league stats (xG etc.):** FBref / Understat, accessible via the `soccerdata`
  Python package. Mind terms of use and rate limits.
- **APIs (optional):** API-Football (detailed stats incl. cards/fouls, freemium),
  football-data.org (thinner, free). Note rate caps.

### 2.2 Research tasks
- Map **each target to a source** that actually contains it (build a small table).
- Understand each schema: column meanings, units, missing-data conventions, encodings.
- Identify **the referee field** (cards/fouls depend heavily on it) — where is it available?
- Understand **how to join/normalize team names** across sources (a real, tedious problem).
- Note **data-leakage hazards** in the schema (e.g. fields only known after kickoff/full time).

---

## Part 3 — Model families (study in this order)

Build understanding incrementally; each model is a baseline for the next.

### 3.1 Baselines (start here — cheap and strong)
- **Elo ratings** (and the football-specific variants for national teams). Research: update
  rule, K-factor, margin-of-victory and home-advantage adjustments. How to convert an Elo
  *difference* into win/draw/loss probabilities.
- **Ordered logit / probit** mapping a strength difference to H/D/A (draw as the middle ordinal category).

### 3.2 The core: Poisson goal models
- **Maher (1982)** — independent Poisson with per-team attack/defense strengths + home advantage.
- **Dixon–Coles (1997)** — the canonical refinement: low-score dependence correction +
  time-decay weighting (recent matches count more). Understand *why* each addition exists.
- Research: how attack/defense parameters are estimated and interpreted.

### 3.3 Extensions
- **Bivariate Poisson** (Karlis–Ntzoufras, 2003) — correlated goals.
- **Negative binomial** goal/count models — for overdispersion.

### 3.4 Bayesian hierarchical models (your sweet spot)
- **Baio & Blangiardo (2010)** — hierarchical Poisson with partial pooling of team strengths;
  implement in PyMC or Stan. This is what handles small national-team samples gracefully and
  gives you full posterior predictive scoreline distributions.

### 3.5 Machine learning (for comparison, and for the count targets)
- **Gradient boosting** (LightGBM / XGBoost / CatBoost): research the **Poisson and Tweedie
  objectives** (use these for goals/cards/corners, not plain regression). Multiclass for 1X2.
- Be ready to find — and report honestly — that a well-tuned Dixon–Coles or Bayesian model
  often beats boosting for 1X2 on match-result data alone. That is a legitimate finding.

### 3.6 Separate models for fouls / cards / corners
- Treat these as their own count models (Poisson/NegBin), **conditioned on the referee** and
  on team style and game state. Consider modelling home/away counts jointly.
- Note: feasible mainly at **league level** where the data exists; for national teams, only
  in tournaments via StatsBomb.

---

## Part 4 — KEYSTONE: from a goal model to the forecasts you want

This is the conceptual center. Research and master the following chain:

1. A goal model gives you two **rate parameters** per match: λ_home, λ_away (expected goals
   for each side, from attack/defense strengths + home advantage).
2. From these you compute the **scoreline probability matrix**: P(home = i, away = j) for
   i, j = 0, 1, 2, … (under independence, the product of two Poisson PMFs; under
   Dixon–Coles, with the low-score correction applied).
3. **Everything else is a sum over that matrix:**
   - P(home win) = sum of cells where i > j; draw where i = j; away win where i < j → **1X2**.
   - Over/Under 2.5 goals = sum where i + j > 2.5, etc.
   - **Correct-score probabilities** are just the individual cells (e.g. cell (2,0) = P(2–0)).

Make sure you can implement this matrix yourself by hand for a single match. Once you can,
scoreline / 1X2 / over-under all come "for free" from one model.

For **fouls/cards/corners**, the analogue is simpler: the count model's predicted mean *is*
the expected count, and its distribution gives you over/under lines for those markets.

---

## Part 5 — The standard pipeline (workflow to follow every time)

1. **Ingest** raw data from a chosen source.
2. **Clean / normalize** — team names, dates, missing values, encodings.
3. **Exploratory analysis** — distributions of goals/fouls/cards/corners; check
   Poisson vs. overdispersion; home-advantage size; referee effects.
4. **Feature engineering** — rolling form, Elo difference, rest days, home/away splits,
   competitive vs. friendly, rolling xG (where available). For national teams, keep it lean.
5. **Fit** the model (start with the baseline, then the structured model).
6. **Derive forecasts** via the Part 4 keystone (scoreline matrix → 1X2 / over-under;
   count models → expected fouls/cards/corners).
7. **Backtest** with strictly **time-ordered, walk-forward** splits (train only on the past).
8. **Evaluate** (Part 6) and **calibrate**.
9. **Compare** models on identical splits and metrics; log results.

> Golden rule: **never shuffle time-series matches into random train/test splits.** Leakage
> here is silent and will make a bad model look excellent.

---

## Part 6 — Evaluation and validation (treat as a first-class part of the project)

Research and implement, in order of importance:

- **Why accuracy is the wrong metric** for probabilistic forecasts.
- **Proper scoring rules:**
  - **Ranked Probability Score (RPS)** — the football-forecasting standard, because H/D/A is
    *ordinal* (rewards putting probability near the true outcome). Study Constantinou & Fenton.
  - **Log loss** (cross-entropy) and **Brier score**.
- **Calibration analysis** — reliability diagrams; are your "20%" events happening ~20% of the time?
- **Count-target metrics** — Poisson deviance; posterior-predictive coverage checks.
- **Benchmarking against the market** — convert bookmaker odds to **de-vigged implied
  probabilities** and treat them as the gold-standard baseline. Beating the closing line is
  extremely hard; closeness is itself a strong, credible result.
- **Walk-forward / rolling-origin backtesting** — research how to set it up correctly.

A clean comparative design for the write-up:
**Elo / ordered-logit baseline → Dixon–Coles → Bayesian hierarchical → gradient boosting**,
all evaluated on the same time-ordered splits with RPS + log loss + calibration.

---

## Part 7 — Phased build plan (what to do first → last)

**Phase 1 — Foundations + first baseline (national teams).**
Resolve Part 0 scope. Refresh Part 1.1–1.3. Acquire national-team results data. Build an
**Elo model**, derive 1X2, backtest with RPS. Goal: end-to-end pipeline working, however simple.

**Phase 2 — Core goal model (national teams).**
Implement **Dixon–Coles**. Build the **scoreline matrix** (Part 4) and produce correct-score,
1X2, and over/under forecasts. Compare against the Elo baseline on RPS + log loss + calibration.

**Phase 3 — Bayesian hierarchical model (national teams).**
Study Part 1.4. Implement Baio–Blangiardo in PyMC/Stan. Exploit partial pooling for sparse
teams. Compare posterior-predictive forecasts to Phases 1–2.

**Phase 4 — Move to leagues + count targets.**
Switch to football-data.co.uk. Re-run the goal models on league data (more matches, cleaner
signal). Add **separate Poisson/NegBin models for fouls, cards, corners**, conditioned on
referee and team style. Benchmark 1X2 against **de-vigged bookmaker odds**.

**Phase 5 — Machine-learning comparison + synthesis.**
Add a gradient-boosting model (Poisson/Tweedie objectives) on engineered features. Compare
all models honestly. Write up: which model wins for which target, and *why* (inductive bias,
data volume, signal strength).

---

## Part 8 — Reading list / references to look up

Foundational papers:
- Maher (1982), *Modelling Association Football Scores*.
- Dixon & Coles (1997), *Modelling Association Football Scores and Inefficiencies in the
  Football Betting Market*.
- Karlis & Ntzoufras (2003), bivariate Poisson models for football.
- Baio & Blangiardo (2010), *Bayesian hierarchical model for the prediction of football results*.
- Constantinou & Fenton (2012), on proper scoring rules (RPS) for football forecasts.
- Hvattum & Arntzen (2010), Elo ratings as a forecasting tool.

Background / tooling:
- Gelman et al., *Bayesian Data Analysis* (hierarchical models, model checking).
- PyMC and/or Stan documentation (hierarchical Poisson examples).
- `statsbombpy`, `soccerdata` package docs; football-data.co.uk column notes.

---

## Part 9 — Decisions and open questions to resolve (keep a running log)

- Which national-team scope: all internationals, or filter to competitive only? Friendlies in or out?
- Time-decay half-life for match weighting (how fast does old form stop mattering)?
- How to encode home advantage for **neutral-venue** tournament matches?
- Poisson vs. negative binomial per target — decide empirically from the EDA.
- How far back to train (data volume vs. relevance trade-off)?
- For cards/fouls: model the referee as a fixed effect or a random effect?
- Where exactly does leakage hide in your chosen schema?

> Every entry here should end with a decision and a one-line justification, so the final
> document defends its choices rather than just presenting results.
