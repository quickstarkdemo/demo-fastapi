FROM python:3.12-slim

# Accept git metadata as build arguments for Datadog source code linking
ARG DD_GIT_REPOSITORY_URL
ARG DD_GIT_COMMIT_SHA

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Consolidate all ENV variables into a single layer
ENV PYTHONPATH=/app \
    PORT=8080 \
    # Observability Provider Selection
    OBSERVABILITY_PROVIDER=datadog \
    # Application Configuration (shared)
    DD_ENV="dev" \
    DD_SERVICE="fastapi-app" \
    DD_VERSION="1.0" \
    # Datadog Configuration
    DD_LOGS_INJECTION=true \
    DD_TRACE_SAMPLE_RATE=1 \
    DD_PROFILING_ENABLED=true \
    DD_DYNAMIC_INSTRUMENTATION_ENABLED=true \
    DD_SYMBOL_DATABASE_UPLOAD_ENABLED=true \
    DD_AGENT_HOST=192.168.1.100 \
    DD_TRACE_AGENT_PORT=8126 \
    DD_DBM_PROPAGATION_MODE=full \
    DD_IAST_ENABLED=true \
    DD_LLMOBS_ENABLED=true \
    DD_LLMOBS_ML_APP=youtube-summarizer \
    DD_LLMOBS_EVALUATORS="ragas_faithfulness,ragas_context_precision,ragas_answer_relevancy" \
    DD_GIT_REPOSITORY_URL=${DD_GIT_REPOSITORY_URL} \
    DD_GIT_COMMIT_SHA=${DD_GIT_COMMIT_SHA} \
    DD_TRACE_HTTP_RESOURCE_PATTERNS_ENABLED=true \
    DD_TRACE_HTTP_RESOURCE_PATTERN="/delete_image/*" \
    DD_CODE_ORIGIN_FOR_SPANS_ENABLED=true \
    DD_EXCEPTION_REPLAY_ENABLED=true \
    # Sentry Configuration (defaults, override at runtime)
    SENTRY_DSN="" \
    SENTRY_ENVIRONMENT="dev" \
    SENTRY_RELEASE="1.0" \
    SENTRY_TRACES_SAMPLE_RATE="1.0" \
    SENTRY_PROFILES_SAMPLE_RATE="1.0" \
    SENTRY_ENABLE_LOGS="true" \
    SENTRY_LOG_BREADCRUMB_LEVEL="INFO" \
    SENTRY_LOG_EVENT_LEVEL="ERROR" \
    SENTRY_SEND_DEFAULT_PII="false" \
    SENTRY_DEBUG="false" \
    SENTRY_ATTACH_STACKTRACE="true"

# Use python -m to run hypercorn
CMD ["python", "-m", "hypercorn", "main:app", "--bind", "0.0.0.0:8080"]

EXPOSE 8080
