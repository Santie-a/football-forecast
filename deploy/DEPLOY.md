# DEPLOY.md — football-forecast on the Pi homelab

Project-specific deploy notes. The host conventions (SSD paths, shared Caddy,
Pi-hole, Tailscale, arm64, no published ports, secrets per-host) come from the
`deploy-to-pi-homeserver` skill — **read that first**; this file only adds what's
specific to this project.

## What actually runs on the Pi

Only the **dashboard** (`app/`): FastAPI + HTMX, read-only. **No** training, MCMC,
or heavy deps run on the Pi — those stay on the PC (see `docs/architecture.md`).
The app image installs only the light dependency group.

```
stack: forecast
  service: app   (FastAPI/uvicorn, joins the `proxy` network, no published port)
  exposed at: http://forecast.homeserver.internal  (via shared Caddy)
```

## The data the Pi needs

The app reads one file: the forecast store
`artifacts/forecasts/forecasts.sqlite`, produced on the PC. On the Pi it lives
under the homelab appdata path and is bind-mounted read-only into the container:

```
/mnt/ssd/appdata/forecast/forecasts.sqlite   ->  /data/forecasts.sqlite (ro)
```

### Refreshing forecasts (PC → Pi)

After running `python -m pipelines.forecast` on the PC:

```bash
# from the PC, over Tailscale (use the Pi's Tailscale name, not a LAN IP):
rsync -av artifacts/forecasts/forecasts.sqlite \
  homeserver:/mnt/ssd/appdata/forecast/forecasts.sqlite
```

`/mnt/ssd` is root-owned — if rsync hits permission errors, stage to a writable
path and `sudo mv` on the Pi, or run the copy as the deploy user. The app opens
the DB read-only, so a live swap is safe; SQLite handles the replace atomically if
you copy to a temp name then `mv`.

A scheduled job can automate this later; manual rsync is fine to start.

### The fixtures store (Pi-owned, writable — do NOT overwrite)

The dashboard's "Manage & queue" page writes upcoming fixtures and results to a
**separate** `fixtures.sqlite`:

```
/mnt/ssd/appdata/forecast/fixtures.sqlite   ->  /data/fixtures.sqlite (rw)
                                                 env: FIXTURES_STORE=/data/fixtures.sqlite
```

Unlike the forecasts store, this file is **owned by the Pi and must never be
rsynced over from the PC** — it holds Pi-side additions. Seed it once on the PC
(`python -m pipelines.wc2026`) and copy it to the Pi a single time, or seed
directly on the Pi. To compute forecasts for queued fixtures on the Pi, sync a
fitted model pickle and run:

```bash
python -m pipelines.process_queue --model-file /data/dixon_coles.pkl \
  --fixtures /data/fixtures.sqlite
```

This is cheap inference (a matrix calc), allowed on the Pi; it never trains.

## First deploy (summary — host steps are in the skill)

1. `ssh homeserver`; `sudo git clone <repo> /mnt/ssd/stacks/forecast` (once a repo exists).
2. `sudo cp app/.env.example app/.env` and fill in (the app needs little — mainly
   the store path, defaulted to `/data/forecasts.sqlite`).
3. Put an initial `forecasts.sqlite` at `/mnt/ssd/appdata/forecast/` (rsync above).
4. `docker compose up -d --build` (builds the arm64 app image on the Pi).
5. Add the Caddy block + Pi-hole record for `forecast.homeserver.internal`
   (see skill → "Exposing a service"); `caddy validate` + `caddy reload`.
6. Verify at `http://forecast.homeserver.internal` from a tailnet device.

## Files to add when implementing the deploy (not yet created)

- `deploy/docker-compose.yml` — the `forecast` stack (app service, `proxy` network,
  bind mount for the store, sane `mem_limit`).
- `app/Dockerfile` — slim python base, install the light dep group, run uvicorn.
- `app/.env.example` — store path + any app config.
- `deploy/Caddyfile.snippet` — the reverse_proxy block to paste into the shared Caddyfile.

Pin image versions (no `:latest`). Keep the app image small — it must not pull
PyMC/LightGBM.
