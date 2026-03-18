# Scripts

Helper scripts for local development and operations. Run them from the repo root (e.g. `./scripts/build.sh`).

## Build & Test
- `build.sh` – Build the container image. Flags: `--local`, `--run`, `--clean`, `--no-cache`, `--podman`, `--rancher`.
- `test.sh` – Pytest wrapper with Datadog CI visibility. Examples: `./scripts/test.sh`, `./scripts/test.sh unit -v`, `./scripts/test.sh fast --cov=src`.

## Environment & Secrets
- `setup-databases.sh` – Interactive helper to load the PostgreSQL/SQL Server schemas from `sql/`.
- `setup-secrets.sh` – Uploads environment variables from a file to GitHub Actions secrets (skips `ENV_FILE_HASH` and `GMKTEC_SSH_KEY` automatically).
- `clear-secrets.sh` – Removes GitHub Actions secrets. Useful flags: `--dry-run`, `--pattern '^SYNOLOGY_'`, `--exclude DD_API_KEY,DD_APP_KEY`, `--all`.

## Deployment & Runners
- `deploy.sh` – SemVer-aware deploy workflow with staged-only git discipline:
  - validates required env keys (`PG*`, `DOCKERHUB_USER`, `DOCKERHUB_TOKEN`)
  - updates `VERSION` via `--bump` / `--version`
  - tracks env sync fingerprints in `.deploy/`
  - generates deploy notes/history and commit metadata
  - can trigger `deploy-self-hosted.yaml` when no staged changes exist
  - flags: `--env-file`, `--version`, `--bump`, `--non-interactive`, `--use-release-brief`, `--ignore-release-brief`
- `setup-runner.sh` – Bootstraps the Docker-based self-hosted GitHub runner using values from `runner.env.example`.

## Observability Utilities
- `setup-sonarqube-monitoring.sh` – Copies the Datadog SonarQube check files into `datadog-conf.d` and can restart the agent.
- `test-sentry.sh` – Verifies Sentry instrumentation against a running container when `OBSERVABILITY_PROVIDER=sentry`.

## Notes
- All scripts expect common tooling (bash, Docker/Podman, Python, `gh` where applicable).
- Run `chmod +x` if your clone did not preserve execute bits.
