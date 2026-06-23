# The models, explained

A from-first-principles tour of every model in this project: the intuition, the math, and *why each one exists* as a fix for the previous one's flaw. Read it top to bottom - each model is a patch on the one before.

Notation: a match has a **home** team and an **away** team. $X$ = home goals, $Y$ = away goals (random variables). $\lambda$ = an expected/mean count. $P(\cdot)$ = probability. Teams are indexed; $\alpha$ = attack strength, $\beta$ = defence strength, $\gamma$ = home advantage.

> The single most important idea: **we don't predict 1X2, over/under, or correct-score directly.** We build *one* model of how many goals each team scores, and *derive* every market from it. Everything below serves that goal.

---

## 0. Why "goals" and why a distribution

We want, for each match, the full table of scoreline probabilities: $P(X = i,\, Y = j)$ for $i, j = 0, 1, 2, \dots$ From that table everything follows (§3). So the whole game is: **model the two goal counts well, with uncertainty.**

"With uncertainty" is the key phrase. A model that says "2–1" is useless for betting/forecasting; a model that says "2–1 is 9%, 1–1 is 11%, …" is what we want. That means we need a *probability distribution over counts*, which leads directly to Poisson.

---

## 1. The Poisson distribution - the atom of everything

A Poisson distribution describes the count of independent events that happen at a constant average rate over a fixed interval (a match). If a team scores at rate $\lambda$ goals/match, the probability it scores exactly $k$ goals is:

\[
P(N = k) = \frac{e^{-\lambda}\,\lambda^{k}}{k!}, \qquad k = 0, 1, 2, \dots
\]

Key property: **mean = variance = $\lambda$**. One parameter controls everything.

**Why it fits goals.** Goals are rare, roughly independent events accumulating over 90 minutes at a fairly stable team-specific rate - exactly Poisson's assumptions. Empirically, goals-per-team-per-match match a Poisson PMF remarkably well. A team with $\lambda = 1.4$ will be shut out $e^{-1.4} \approx 25\%$ of the time, score once $\approx 34\%$, twice $\approx 24\%$, etc.

**Where it breaks - overdispersion.** Poisson forces variance = mean. Many counts in football have *variance > mean* (more spread than Poisson allows): **fouls and cards** especially (a single bad-tempered match produces a cluster of cards). When you see variance $\gg$ mean in the EDA, Poisson is wrong and you reach for the **negative binomial** (§7). Goals are usually fine; cards/fouls usually are not.

**Zero-inflation.** Some counts have *more zeros* than even their rate implies - **red cards** are mostly zero. That's a separate failure mode (zero-inflated models), relevant only for the count targets later.

---

## 2. Elo - the cheap, strong baseline

Before any goal model, Elo gives a single **strength rating** per team and a way to turn a rating *difference* into win/draw/loss probabilities. It's a baseline you must beat.

**Expected result from a rating difference.** If home rating $R_h$, away rating $R_a$, the model's expected score for home (on a 0–1 scale, win = 1, draw = 0.5) is the logistic:

\[
E_{\text{home}} = \frac{1}{1 + 10^{-(R_h + H - R_a)/400}}
\]

$H$ = a home-advantage rating bump (0 on neutral venues - decision M03). The $/400$ and base-10 are conventions; they set how much a rating gap means.

**The update rule.** After a match with actual result $S$ (1 / 0.5 / 0 for home), move ratings toward the surprise:

\[
R_h \leftarrow R_h + K\,(S - E_{\text{home}}), \qquad R_a \leftarrow R_a - K\,(S - E_{\text{home}})
\]

$K$ = learning rate (how fast ratings react). Big $K$ = jumpy, reactive; small $K$ = stable, slow. Beating an underdog gains little; beating a favourite gains a lot - that's the $(S - E)$ surprise term.

**Football refinements:**
- **Margin of victory (MOV):** a 5–0 should move ratings more than a 1–0. Scale $K$ by a function of goal difference (the FIFA/World-Football-Elo approach).
- **Home advantage $H$:** a fixed rating bump, zeroed for neutral venues.

**Getting 1X2 out of Elo.** Elo natively gives win-vs-not. To split out **draws** you need an extra step - e.g. map the rating difference through an **ordered logit** (§2.1) or an empirical draw model, since Elo alone has no notion of a draw probability. This is Elo's main weakness for our task, and the reason we move to goal models.

### 2.1 Ordered logit/probit - strength difference → H/D/A

Treat the outcome as **ordinal** (away-win < draw < home-win) and model it with two thresholds on a latent "home superiority" axis driven by the strength difference. It's a principled way to produce three calibrated probabilities from one number, and a clean baseline alongside Elo. But it still throws away the scoreline - which the Poisson models keep.

---

## 3. The keystone: from rates to a scoreline matrix

This is the conceptual centre of the whole project. Suppose a goal model hands you two rates for a match: $\lambda_{\text{home}}$ and $\lambda_{\text{away}}$ (expected goals for each side). Assume (for now) the two counts are independent Poissons. Then every cell of the scoreline table is just a product:

\[
P(X = i,\, Y = j) = \text{Poisson}(i;\lambda_{\text{home}})\cdot\text{Poisson}(j;\lambda_{\text{away}}) = \frac{e^{-\lambda_{\text{home}}}\lambda_{\text{home}}^{\,i}}{i!}\cdot\frac{e^{-\lambda_{\text{away}}}\lambda_{\text{away}}^{\,j}}{j!}
\]

Compute this for $i, j = 0 \dots$ (say) $10$ and you have the *scoreline matrix* $M$, where $M[i][j] = P(i\text{–}j)$. **Everything is a sum over $M$:**

\[
\begin{aligned}
P(\text{home win}) &= \sum_{i > j} M[i][j] \\
P(\text{draw}) &= \sum_{i = j} M[i][j] \qquad (\text{this is your 1X2}) \\
P(\text{away win}) &= \sum_{i < j} M[i][j] \\
P(\text{over }2.5) &= \sum_{i + j \ge 3} M[i][j] \\
P(2\text{–}0) &= M[2][0] \qquad (\text{correct-score is a single cell})
\end{aligned}
\]

Master this once (it's a ~10-line function in `forecast/`) and scoreline, 1X2, over/under, BTTS, correct-score all come **for free** from any model that produces $\lambda_{\text{home}}, \lambda_{\text{away}}$. The rest of the modelling effort is just: *get those two rates right.*

---

## 4. Maher (1982) - the first real goal model

Elo gives one number per team; goals need *two* skills: scoring and conceding. Maher gives every team an **attack** strength $\alpha_i$ and a **defence** strength $\beta_i$, plus a global **home advantage** $\gamma$. The rates for a match between home $i$ and away $j$:

\[
\lambda_{\text{home}} = \alpha_i\,\beta_j\,\gamma, \qquad \lambda_{\text{away}} = \alpha_j\,\beta_i
\]

(Equivalently written additively in log-space: $\log\lambda_{\text{home}} = a_i - d_j + h$, which is how you actually fit it - as a Poisson regression / GLM.)

**How it's fit - maximum likelihood.** Over a season of $N$ matches, assume each match's two goal counts are independent Poissons with the rates above. The likelihood is the product over all matches of the two Poisson PMFs; you maximize its log:

\[
\log L = \sum_{\text{matches}} \big[\, x\log\lambda_{\text{home}} - \lambda_{\text{home}} + y\log\lambda_{\text{away}} - \lambda_{\text{away}} + \text{const} \,\big]
\]

Optimizing over all $\{\alpha_i, \beta_i, \gamma\}$ recovers each team's attack/defence and the home edge. (One identifiability constraint is needed, e.g. fix the mean attack to 1, because scaling all attacks up and defences down leaves $\lambda$ unchanged.)

**What's still wrong:** (a) it treats home and away goals as **independent** - false at low scores; (b) it weights a match from 5 years ago the same as last week. Dixon–Coles fixes both.

---

## 5. Dixon–Coles (1997) - the canonical refinement

Two targeted patches on Maher. This is the workhorse; most production models are a flavour of it.

**Patch 1 - low-score dependence.** Real football has *more* 0–0 and 1–1 draws and *fewer* 1–0 / 0–1 than independent Poisson predicts (teams "kill the game" at 0–0; a leading team defends). Dixon–Coles multiplies the four low-score cells by a correction factor $\tau$ (tau) controlled by one parameter $\rho$ (rho):

\[
P(X = i,\, Y = j) = \tau(i, j;\lambda_{\text{home}}, \lambda_{\text{away}}, \rho)\cdot\text{Poisson}(i;\lambda_{\text{home}})\cdot\text{Poisson}(j;\lambda_{\text{away}})
\]

where $\tau$ adjusts only the $(0,0), (0,1), (1,0), (1,1)$ cells ($= 1$ everywhere else). $\rho < 0$ inflates draws - exactly the observed bias. It's a minimal, surgical fix: the independence assumption stays everywhere except where it demonstrably fails.

**Patch 2 - time decay.** Recent form matters more. Down-weight each match in the likelihood by its age with an exponential decay:

\[
w(t) = e^{-\xi\,\Delta t}, \qquad \Delta t = \text{days before the match being predicted}
\]

$\xi$ (xi) sets the **half-life** (how fast old matches stop mattering). This is a hyperparameter we tune by backtest RPS (decision M02). The likelihood becomes a *weighted* sum of the per-match log-likelihoods.

After fitting, you read off $\lambda_{\text{home}}, \lambda_{\text{away}}$ for any fixture and run the keystone (§3) - now with the $\tau$ correction baked into the matrix.

---

## 6. Bivariate Poisson - correlation, done properly

Dixon-Coles patches correlation only at low scores via a fudge factor. The **bivariate Poisson** (Karlis-Ntzoufras 2003) models it structurally. Construct:

\[
X = W_1 + W_3, \qquad Y = W_2 + W_3
\]

with $W_1, W_2, W_3$ independent Poissons (rates $\lambda_1, \lambda_2, \lambda_3$). The **shared** component $W_3$ (rate $\lambda_3$) is common to both teams, inducing $\operatorname{Cov}(X, Y) = \lambda_3 \ge 0$. So $\lambda_3$ *is* the correlation knob: 0 collapses back to independent Poisson. Interpretation: shared match conditions (an open end-to-end game, weather, referee letting play flow) lift both teams' goals together.

Limitation: this construction only produces **non-negative** correlation. Football's low-score effect is a slight *negative* dependence, which is why Dixon–Coles's $\tau$ hasn't been fully retired. In practice people use one or the other; we'll compare.

---

## 7. Negative binomial - when variance beats the mean

Poisson's mean = variance is a straitjacket. The **negative binomial (NB)** adds a dispersion parameter that lets variance exceed the mean. One clean way to see it: NB is a **Poisson whose rate $\lambda$ is itself random** (Gamma-distributed). The extra randomness in $\lambda$ inflates the variance:

\[
\operatorname{Var} = \mu + \frac{\mu^{2}}{r}
\]

($r \to \infty$ recovers Poisson; small $r$ = heavy overdispersion.)

**Where we use it:** the **count targets** - fouls and cards - which the EDA will likely show are overdispersed (decision M04, decided empirically). Goals usually stay Poisson. Same keystone machinery applies: NB gives a count distribution, its mean is the "expected fouls/cards", and its tail gives over/under lines.

---

## 8. Bayesian hierarchical models - the sweet spot for sparse teams

The problem with national teams: some play 6 matches a year. Fitting an independent attack/defence for a rarely-seen team by MLE gives wild, overconfident estimates (small-sample noise). Bayesian **hierarchical / partial-pooling** models fix this elegantly.

**The idea.** Don't estimate each team in isolation, and don't lump everyone together either — do **both, weighted by evidence**. Assume every team's attack strength is drawn from a *shared league-wide distribution*:

\[
\begin{aligned}
\alpha_i &\sim \mathcal{N}(\mu_{\text{att}}, \sigma_{\text{att}}) \\
\beta_i &\sim \mathcal{N}(\mu_{\text{def}}, \sigma_{\text{def}}) \\
\log\lambda_{\text{home}} &= \mu + \alpha_{\text{home}} - \beta_{\text{away}} + \gamma \qquad (\text{and the away analogue})
\end{aligned}
\]

Now a team with little data gets **shrunk toward the population mean** $\mu_{\text{att}}$ (it "borrows strength" from everyone else); a team with lots of data is allowed to sit far from the mean because its own evidence dominates. The amount of shrinkage is *learned* from the data via $\sigma_{\text{att}}$, not hand-set. This is exactly what makes sparse-sample national-team estimates sane (Baio & Blangiardo 2010).

**Prior $\rightarrow$ posterior $\rightarrow$ posterior predictive.** Bayesian inference gives you a full **posterior** distribution over all parameters, not a point estimate. For forecasting we want the **posterior predictive**: integrate the scoreline probabilities over the whole posterior, so the forecast *automatically includes parameter uncertainty*. A rarely-seen team correctly comes out with *wider*, humbler scoreline distributions, a property MLE models can't express.

**How it's computed - MCMC.** These posteriors have no closed form, so we sample them with **Markov-chain Monte Carlo** (via PyMC). Practical literacy you need:
- **Chains & draws:** run several chains; each is a random walk that, after "burn-in", samples from the posterior.
- **R-hat (convergence):** compares within- vs between-chain variance; want $\hat{R} \approx 1.00$. Values $> 1.01$ mean the chains disagree; don't trust the fit.
- **Trace plots:** healthy traces look like "fuzzy caterpillars" (well-mixed); trends or stuck regions signal trouble.
- **Effective sample size, divergences:** sanity checks PyMC reports.

This is the **PC-only** step — MCMC is the heaviest compute and won't run usefully on the Pi (architecture decision 001).

---

## 9. Gradient boosting - the ML challenger

A different philosophy: forget the goal-generating story, throw **engineered features** (Elo diff, rolling form, rest days, home/away splits, rolling xG) at a gradient-boosted tree ensemble and let it learn patterns. Trees are built sequentially, each correcting the previous ensemble's errors.

**The crucial detail — use the right objective.** For counts, don't use plain squared-error regression. Use the **Poisson** or **Tweedie** objective (LightGBM/XGBoost support both), which model the target as a count/non-negative quantity with the right error structure. For 1X2, use **multiclass** (softmax) to get three calibrated probabilities.

**Honest expectation.** On match-result data alone, a well-tuned Dixon–Coles or Bayesian model **often beats boosting** for 1X2 — the structured goal model encodes the right inductive bias and needs less data. Boosting tends to shine where it can exploit many features (the **count targets** at league level, with referee/style signals). Reporting "the simple structured model won" is a legitimate, valuable finding — not a failure.

---

## 10. The count targets - fouls, cards, corners

Treated as their **own** count models (Poisson if not overdispersed, else NB; §1, §7), one per target, conditioned on:
- the **referee** (huge for cards/fouls - some refs card twice as much),
- team **style** (some teams foul more),
- **game state** (a tight derby vs a dead rubber).

Model home and away counts, possibly jointly. **Feasible mainly at league level** (Phase 4) where the data exists; for national teams only major tournaments have it (StatsBomb). The output mean = "expected fouls/cards/corners"; the distribution gives over/under markets. Same keystone philosophy, simpler (no opponent-coupling matrix needed unless you model the two sides jointly).

---

## 11. Evaluation - how we know a model is good

You cannot judge probabilistic forecasts by accuracy. Picking the most likely 1X2 and counting hits ignores *how confident* the model was and *how close* the probabilities were - the whole point.

**Proper scoring rules** reward honest, well-calibrated probabilities (you can't game them by shading your numbers):

- **RPS (Ranked Probability Score)** - *the* football metric, because H/D/A is **ordinal**. It penalizes the *cumulative* distance between predicted and actual outcome, so predicting "draw" when the truth is "home win" hurts less than predicting "away win". With $C$ outcome categories, predicted cumulative probabilities $p_i$ and the cumulative outcome indicator $o_i$:

\[
\text{RPS} = \frac{1}{C - 1}\sum_{k=1}^{C-1}\left(\sum_{i=1}^{k}(p_i - o_i)\right)^{2} \qquad (\text{lower = better})
\]

  (Constantinou & Fenton 2012.) This is the primary number we optimize.
- **Log loss** (cross-entropy): $-\sum_i o_i \log p_i$. Punishes confident wrong calls harshly (a 1% on the true outcome $\rightarrow$ huge penalty). Secondary.
- **Brier score:** mean squared error of the probability vector. Secondary.

**Calibration vs sharpness.** *Calibration*: do events you call "20%" happen ~20% of the time? Check with a **reliability diagram** (predicted prob bucket vs observed frequency - want the diagonal). *Sharpness*: how confident (far from base rates) you dare to be. The goal is the **sharpest** forecast that **stays calibrated**. A model can be perfectly calibrated and useless (always predicts base rates), so you need both lenses.

**The market benchmark.** Bookmaker odds, converted to probabilities and **de-vigged** (the bookmaker's margin removed so the three probabilities sum to 1), are the gold-standard baseline - they encode enormous information. Beating the **closing line** is extremely hard; getting *close* is itself a strong, credible result. Always report your models against this baseline, not just against each other.

**Count targets:** Poisson deviance, and posterior-predictive coverage checks (do the realized counts fall inside your predicted intervals at the right rate?).

---

## 12. How it all fits together (build order)

Each model is the baseline for the next, evaluated on **identical, time-ordered, walk-forward splits** with RPS + log loss + calibration. **Never shuffle matches into random splits** - temporal leakage silently makes bad models look great.

```
Elo / ordered-logit          ->  cheap baseline, 1X2 only (no scoreline)
        │  add: per-team attack/defence + a goal distribution
        ▼
Maher (independent Poisson)  ->  first real scoreline matrix
        │  add: low-score τ correction + time decay
        ▼
Dixon–Coles                  ->  the workhorse; correct 1X2/scoreline/O-U/CS
        │  add: structural correlation  /  overdispersion  /  full uncertainty
        ▼
Bivariate Poisson · NegBin · Bayesian hierarchical
        │  alternative philosophy: features + trees
        ▼
Gradient boosting            ->  shines on count targets; compare honestly
```

The thread through all of it: **estimate goal rates $\rightarrow$ build the scoreline matrix (§3) $\rightarrow$ sum it into the markets $\rightarrow$ score with RPS.** Get those two rates right and everything else is arithmetic.

---

## Where to go deeper

The primary sources (roadmap Part 8): Maher (1982); Dixon & Coles (1997); Karlis & Ntzoufras (2003); Baio & Blangiardo (2010); Constantinou & Fenton (2012); Hvattum & Arntzen (2010, Elo). For the Bayesian machinery: Gelman et al., *Bayesian Data Analysis*, and the PyMC docs' hierarchical-Poisson examples. Write your own ½–1 page summary of each in `research/notes/` — if you can't reproduce the math without notes, you don't own it yet.
