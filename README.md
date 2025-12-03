# FastAPI Image & Video Service

FastAPI app for image ingestion (S3 + Rekognition + Mongo/Postgres/SQL Server storage) and OpenAI-powered YouTube summarization with optional Notion persistence. Observability is pluggable: Datadog by default, Sentry or a noop provider when needed.

## Quick Start
- Requirements: Python 3.12+, Docker (optional), AWS credentials for S3/Rekognition/SES, OpenAI API key, optional Mongo/Postgres/SQL Server, optional Datadog or Sentry credentials.
- Install deps:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp env.example .env
  ```
- Set the minimum env needed for the features you plan to use:
  - Observability: `OBSERVABILITY_PROVIDER=datadog|sentry|disabled`
  - OpenAI: `OPENAI_API_KEY`
  - AWS: `AMAZON_KEY_ID`, `AMAZON_KEY_SECRET`, `AMAZON_S3_BUCKET`, `SES_REGION`, `SES_FROM_EMAIL`
  - Databases (pick what you use): `MONGO_CONN`/`MONGO_USER`/`MONGO_PW`, `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGPASSWORD`, `SQLSERVERHOST`/`SQLSERVERPORT`/`SQLSERVERDB`/`SQLSERVERUSER`/`SQLSERVERPW`, plus `SQLSERVER_ENABLED=true|false`
  - Optional: `NOTION_API_KEY`, `NOTION_DATABASE_ID`, `DD_API_KEY`, `DD_APP_KEY`, `BUG_REPORT_EMAIL`
- Run the app (mirrors the Docker entrypoint):
  ```bash
  python -m hypercorn main:app --bind 0.0.0.0:8080
  # or uvicorn main:app --reload
  ```
- Docker:
  ```bash
  docker build -t fastapi-app .
  docker run -p 8080:8080 --env-file .env fastapi-app
  ```
- Docs live at `http://localhost:8080/docs` and `http://localhost:8080/redoc`.

## API Map
- **Images**
  - `GET /images?backend=mongo|postgres|sqlserver` – List stored images.
  - `POST /add_image?backend=mongo|postgres|sqlserver` – Upload an image (S3 + Rekognition) and persist metadata.
  - `DELETE /delete_image/{id}?backend=...` – Remove image and metadata.
- **Amazon S3**
  - `POST /api/v1/upload-image-amazon/` – Upload a file directly to S3.
  - `DELETE /api/v1/delete-one-s3/{key}` – Delete a single object.
  - `DELETE /api/v1/delete-all-s3` – Delete all objects in the configured bucket.
- **Database-specific helpers**
  - `GET /api/v1/mongo/get-image-mongo/{id}`
  - `DELETE /api/v1/mongo/delete-all-mongo/{key}`
  - `GET /api/v1/postgres/get-image-postgres/{id}`
  - `GET /api/v1/sqlserver/get-image-sqlserver/{id}`
  - `GET /api/v1/database-status` and `GET /api/v1/database-config` – Live backend status/config snapshots.
- **OpenAI & Notion**
  - `GET /api/v1/openai-hello` – Sanity check.
  - `GET /api/v1/openai-gen-image/{search}` – DALL·E 3 image generation.
  - `POST /api/v1/summarize-youtube` – Summarize a single video (see `examples/youtube_batch_usage.py` for payloads).
  - `POST /api/v1/batch-summarize-youtube` – Batch processing strategies for multiple URLs.
  - `POST /api/v1/save-youtube-to-notion` – Persist summaries to a Notion database.
- **Datadog utilities**
  - `GET /datadog-hello`, `POST /datadog-event`, `GET /datadog-events`
  - `POST /app-event/{event_type}`, `POST /track-api-request`, `POST /bug-detection-event`
- **Diagnostics & General**
  - `GET /health` – Includes service/env/version and observability provider info.
  - `GET /test-sqlserver` – Debug the SQL Server connection.
  - `GET /timeout-test?timeout=N` – Force a slow response for profiling.
  - `GET /test-sentry-logs`, `GET /sentry-diagnostics` – Sentry-only debugging endpoints.
  - `POST /create_post` – Sample JSONPlaceholder proxy.
  - `GET /` – Basic welcome message.

## Configuration Cheatsheet
- Observability is set via `OBSERVABILITY_PROVIDER` (`datadog` default, `sentry`, or `disabled`). Datadog tags use `DD_SERVICE`, `DD_ENV`, `DD_VERSION`. Sentry mirrors these with `SENTRY_*`.
- AWS/SES, database, and OpenAI credentials are read from `.env` at startup. The app loads `.env` from the repo root (`APP_ROOT/.env`).
- See `env.example` for every option, including profiling and Datadog DBM flags.

## Development & Testing
- Run tests with Datadog visibility: `./scripts/test.sh fast` or `./scripts/test.sh unit -v`.
- Direct pytest works too: `pytest`, `pytest --cov=src`.
- Helpful scripts live in `scripts/README.md`:
  - `build.sh` for container builds, `test.sh` for tests.
  - `setup-databases.sh` to load SQL schemas.
  - `setup-secrets.sh` / `clear-secrets.sh` for GitHub Actions secrets.
  - `deploy.sh` for the guided deployment flow.
  - `test-sentry.sh` and `setup-sonarqube-monitoring.sh` for observability checks.

## Deployment Notes
- The Docker image uses Hypercorn with `CMD ["python", "-m", "hypercorn", "main:app", "--bind", "0.0.0.0:8080"]`.
- GitHub Actions workflows live in `.github/workflows/` (self-hosted and GitHub-hosted variants).
- For manual pushes to production or secret refresh, use `./scripts/deploy.sh [env-file] [--force]`.

## Project Layout
- Application entrypoint: `main.py`
- Core services: `src/amazon.py`, `src/openai_service.py`, `src/datadog.py`, `src/mongo.py`, `src/postgres.py`, `src/sqlserver.py`, `src/database_status.py`
- Observability providers: `src/observability/`
- SQL schemas: `sql/`
- Examples: `examples/youtube_batch_usage.py`
- Docs: `docs/`
- Tooling: `scripts/`, `.github/workflows/`, `Dockerfile`
