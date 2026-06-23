# Implementation plan

The roadmap is a *research* plan (what to learn, in what order). This is the *dev* plan: the concrete modules, the stable contracts they share, the build sequence, and the acceptance criteria that say a piece is done. Read it alongside [`roadmap.md`](../roadmap.md) (the why) and [`docs/models-explained.md`](models-explained.md) (the math).

Guiding principle: **define a handful of stable interfaces once, then every model is a swappable implementation behind them.** A new model should mean one new file in `models/` and zero changes to the backtester, the keystone, or the store.

---

## 1. The contracts (build these first, freeze them early)

These five contracts are the spine. Phases add models *behind* them; they should rarely change.

### 1.1 The canonical match record (`data/schema.py`)

One row per match, one schema across all sources (national + league). Optional columns stay null until a source provides them. This is what every model and the backtester consume — no model reads a raw CSV.

| column | type | phase | notes |
|--------|------|-------|-------|
| `match_id` | str | 1 | stable, source-independent id |
| `date` | date | 1 | kickoff day (drives time ordering) |
| `competition` | str | 1 | e.g. "FIFA World Cup", "Premier League" |
| `comp_type` | enum | 1 | `friendly` / `qualifier` / `final_tournament` / `league` / `continental` — the M01 feature |
| `home`, `away` | str | 1 | canonical team names |
| `home_goals`, `away_goals` | int | 1 | full-time goals |
| `neutral` | bool | 1 | drives M03 (no home advantage if true) |
| `season` | str | 2 | for league phases |
| `referee` | str | 4 | cards/fouls driver |
| `home_fouls`, `away_fouls`, `*_cards`, `*_corners` | int | 4 | count targets |
| `odds_h`, `odds_d`, `odds_a` | float | 4 | for the de-vigged market benchmark |

A `validate(df) -> df` function enforces dtypes, non-null required columns, and `home != away`. **Leakage rule encoded here:** only columns known *before* kickoff may be used as features; goals/cards/etc. are targets, never features for the same match.

### 1.2 The keystone (`forecast/scoreline.py`, `forecast/markets.py`)

The single most reused code. Pure functions, no model state.

```python
def scoreline_matrix(lambda_home: float, lambda_away: float,
                     max_goals: int = 10, dc_rho: float | None = None) -> np.ndarray:
    """P(i,j) matrix. dc_rho=None → independent Poisson; else Dixon–Coles tau correction."""

def one_x_two(M: np.ndarray) -> dict:        # {"H":.., "D":.., "A":..}
def over_under(M: np.ndarray, line: float) -> dict
def correct_score(M: np.ndarray) -> dict      # {(i,j): p}
def btts(M: np.ndarray) -> dict
```

### 1.3 The model protocol (`models/base.py`)

Two protocols so result-only models (Elo) and goal models (Dixon–Coles, Bayesian) share one shape:

```python
class OutcomeModel(Protocol):
    def fit(self, matches: pd.DataFrame, asof: date) -> "OutcomeModel": ...
    def predict_1x2(self, fixture: Fixture) -> dict: ...   # {"H","D","A"}

class GoalModel(OutcomeModel, Protocol):
    def rates(self, fixture: Fixture) -> tuple[float, float]:  # (lambda_home, lambda_away)
        ...
    # predict_1x2 is derived for free: one_x_two(scoreline_matrix(*self.rates(fixture)))
```

`fit(..., asof)` is mandatory and central to the no-leakage rule: a model may only see matches with `date < asof`. The backtester enforces it; models must honour it.

### 1.4 The walk-forward backtester (`eval/backtest.py`)

Model-agnostic. Takes a *factory* (so it refits cleanly at each step), the data, and a split schedule; returns a tidy metrics frame. **Never** does a random split.

```python
def walk_forward(model_factory: Callable[[], OutcomeModel],
                 matches: pd.DataFrame,
                 schedule: SplitSchedule,    # e.g. expanding window, monthly origins
                 metrics: list[Metric]) -> pd.DataFrame:  # model × origin × metric
```

### 1.5 The store (`store/forecasts.py`, `store/metrics_store.py`)

Read/write the artifact store exactly per [`docs/data.md`](data.md#artifact-store-schema). The PC writes; the app reads. The app depends only on this schema, never on `models/`.

```python
def write_forecasts(df: pd.DataFrame, path="artifacts/forecasts/forecasts.sqlite") -> None
def read_forecasts(filters=...) -> pd.DataFrame
def write_backtests(df, path="artifacts/metrics/backtests.parquet") -> None
```

---

## 2. Cross-cutting conventions

- **CLI:** each `pipelines/*.py` is thin — parse args, call the library, write an artifact. Standard flags: `--asof YYYY-MM-DD`, `--model`, `--seed`, `--config`.
- **Config + seeds:** every run takes a config (dataclass or TOML) and a seed, and **writes them next to its artifact** so any output is traceable (Elo has no seed; MCMC/boosting do).
- **Testing:** keystone and metrics get **golden-number** unit tests (hand-computed small matrices / known RPS values). Each model gets a fit/predict smoke test on a tiny synthetic dataset. The backtester gets a leakage test (assert no future match leaks into a fit).
- **Reproducibility:** per [`workflow.md`](workflow.md#reproducing-a-phases-results) — every phase ends with a reproduce note (commands, seed, expected metrics) in its `research/notes/` write-up.
- **Definition of done (every phase):** tests pass · decision(s) logged in [`decisions.md`](decisions.md) · backtest metrics table updated · reproduce note written · pipeline still runnable end-to-end.

---

## 3. Build sequence

Each phase keeps the **whole pipeline runnable**; models get better, the harness doesn't change. The dashboard is a **parallel track** that can start the moment Phase 1 writes the store.

### Phase 1 — End-to-end skeleton with Elo  *(national teams)*
**Goal:** the thinnest possible ingest → model → backtest → store loop, working.
- `data/schema.py` + `data/sources/intl_results.py` (load + normalize the international results dataset) + `data/teamnames.py`.
- `models/elo.py` (update rule, MOV, home-advantage with M03 neutral handling) + a draw model / `models/ordered_logit.py` to get H/D/A.
- `eval/metrics.py` (RPS, log loss) + `eval/backtest.py` (expanding-window walk-forward).
- `store/forecasts.py` (write) + `pipelines/{ingest,train,backtest,forecast}.py`.
- **Acceptance:** `python -m pipelines.backtest --model elo` runs on real data and **beats a base-rate baseline** (predicting the historical H/D/A frequencies) on RPS. Calibration sane. Reproduce note written.

### Phase 2 — Core goal model: Dixon–Coles
**Goal:** real scoreline distributions; 1X2/OU/CS derived via the keystone.
- `forecast/scoreline.py` + `forecast/markets.py` (full keystone, incl. DC `tau`).
- `models/maher.py` then `models/dixon_coles.py` (MLE fit with time-decay weighting; `rates()` + derived `predict_1x2`).
- Add Brier + calibration (reliability diagram) to `eval/`.
- **Acceptance:** Dixon–Coles **beats Elo** on RPS *and* log loss on identical splits; scoreline matrix validated against a hand-computed example (golden test); time-decay half-life selected by backtest (resolves M02).

### Phase 3 — Bayesian hierarchical (PyMC)  *(PC-only; needs C toolchain)*
**Goal:** partial pooling + posterior-predictive scorelines for sparse teams.
- `models/bayesian.py` (Baio–Blangiardo in PyMC); posterior-predictive → scoreline matrix → store.
- Convergence gating in the fit (`R-hat`, divergences) — refuse to emit forecasts from a non-converged fit.
- **Acceptance:** `R-hat < 1.01`; matches/beats Dixon–Coles on RPS; **demonstrably wider intervals for low-data teams** (the whole point). Install the C compiler first (decision 006 note).

### Phase 4 — Leagues + count targets
**Goal:** move to data where fouls/cards/corners exist; benchmark against the market.
- `data/sources/football_data_couk.py` (incl. referee + odds); extend schema usage.
- `models/counts.py` (Poisson vs NegBin per target — resolves M04 from EDA; referee effect — resolves M06).
- `eval/market.py` (de-vig odds → implied probabilities baseline).
- **Acceptance:** count models calibrated (posterior-predictive coverage); 1X2 reported **against de-vigged closing odds**; leakage catalogue (M07) populated for this schema.

### Phase 5 — Gradient boosting + synthesis
**Goal:** the ML challenger and an honest write-up.
- `features/` (Elo diff, rolling form, rest days, home/away splits, rolling xG) + `models/boosting.py` (Poisson/Tweedie for counts, multiclass for 1X2).
- **Acceptance:** all models on one comparison table (RPS/log loss/calibration, vs market); written conclusion on *which model wins which target and why*.

### App track (parallel, starts after Phase 1)
**Goal:** the deployable visualization — your stated end goal — validated early against the store contract.
- `app/main.py` + templates: fixtures list, match detail (scoreline heatmap, 1X2 bars, O/U, expected counts once Phase 4 lands), model-comparison page.
- Reads the store **read-only**; no modelling deps.
- Deploy to the Pi per [`deploy/DEPLOY.md`](../deploy/DEPLOY.md) once the match-detail page renders a real forecast. Doing this right after Phase 1 proves the PC→store→Pi pipeline end-to-end while the models are still simple.

---

## 4. First concrete tasks (Phase 1, ready to code)

1. `data/schema.py` — columns + `validate()`; a `Fixture` type (a not-yet-played match).
2. `data/sources/intl_results.py` — download/load the international results CSV → canonical schema; unit test on a small sample.
3. `eval/metrics.py` — `rps()`, `log_loss()` with golden-number tests.
4. `models/elo.py` — `EloModel(fit/predict_1x2)` honouring `asof` and M03.
5. `eval/backtest.py` — `walk_forward()` + the leakage test.
6. `store/forecasts.py` — `write_forecasts()` matching the data.md schema.
7. `pipelines/{ingest,train,backtest,forecast}.py` — wire 1–6 together.
8. Reproduce note + decisions update + a base-rate baseline to beat.

> When we start Phase 1, the smallest first PR is items 1 + 3 (schema + metrics with golden tests) — pure functions, no data dependency, immediately testable.
