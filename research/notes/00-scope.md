# 00 — Scope and framing (roadmap Part 0)

Seed write-up of the up-front framing decisions. Expand each section to ½–1 page
in your own words as you internalize it (roadmap's "explain it without notes" rule).

## Goals vs. expected goals (xG)

**xG is an input, not the output.** xG is a *shot-quality metric*: given a shot's
location, angle, type, etc., it estimates the probability that shot becomes a goal.
It summarizes chance quality. **We forecast *goals scored*** — the actual count a
team will register — as a model *output*. xG (where available) is therefore a
*feature* that may help predict goals; it is never the thing we are predicting.
Conflating them would mean "predicting a prediction."

## What "expected fouls / cards / corners" means

The **mean of a predicted count distribution**, *and* we keep the full distribution.
The mean answers "how many corners on average"; the distribution answers over/under
market questions ("P(corners > 9.5)"). So each count target is a full count model
(Poisson/NegBin), and "expected X" = its mean. (roadmap Part 4 analogue.)

## National teams vs. clubs — why it changes the project

- **Small samples:** few matches/year, long gaps, rosters change → sparse data per
  team. This is why partial pooling / hierarchical models matter, and why we keep
  friendlies (decision M01) rather than discard scarce matches.
- **Match type matters:** friendlies vs. qualifiers vs. finals differ in intensity
  and effort → encode competition type as a feature (M01).
- **Data asymmetry:** scorelines exist for ~150 years of internationals, but
  per-match fouls/cards/corners are scarce for national teams (only major
  tournaments, via StatsBomb). They are abundant at league level.

## Sequencing consequence

Build **scoreline + 1X2 for national teams first** (World Cup as the first concrete
validation target — decision 007). Add **fouls/cards/corners count models when
moving to leagues** (Phase 4), where the data to fit and validate them exists.
Keep all code **competition-agnostic** so the same pipeline later serves leagues,
continental, and club competitions.

## Decisions captured

See `docs/decisions.md`: M01 (all internationals + type feature), M03 (no home
advantage on neutral venues), 007 (competition-agnostic core, World Cup first).
Deferred-to-EDA: M02, M04, M05 (decay/window/distribution), M06 (referee, Phase 4).
