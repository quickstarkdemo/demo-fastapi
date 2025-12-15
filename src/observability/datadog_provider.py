"""
Datadog observability provider.

Wraps Datadog's ddtrace library to implement the ObservabilityProvider interface.
Maintains all existing Datadog functionality while providing a consistent API.
"""

import os
import logging
import traceback
from typing import Any, Callable, Optional, Dict, List, ContextManager
from contextlib import contextmanager

from .base import ObservabilityProvider

logger = logging.getLogger(__name__)


class DatadogProvider(ObservabilityProvider):
    """
    Datadog observability provider using ddtrace.

    Implements comprehensive tracing, profiling, and error tracking via Datadog APM.
    Requires DD_AGENT_HOST and DD_TRACE_AGENT_PORT to be configured.
    """

    def __init__(self):
        """Initialize the Datadog provider."""
        self._initialized = False
        self._tracer = None
        self._enabled = self._check_configuration()
        logger.info(f"DatadogProvider initialized - enabled: {self._enabled}")

    def _check_configuration(self) -> bool:
        """
        Validate that required Datadog configuration is present.

        Returns:
            True if Datadog is properly configured, False otherwise
        """
        # Check for required configuration (API keys for events are optional)
        dd_agent_host = os.getenv('DD_AGENT_HOST')
        if not dd_agent_host:
            logger.warning("DD_AGENT_HOST not configured - Datadog traces will not be sent")
            return False

        return True

    def initialize(self) -> None:
        """
        Initialize Datadog tracing and instrumentation.

        Sets up:
        - Automatic instrumentation via patch_all()
        - Runtime metrics collection
        - Profiler (if enabled)
        - LLM Observability (if enabled)
        """
        if self._initialized:
            logger.warning("DatadogProvider already initialized")
            return

        if not self._enabled:
            logger.warning("Datadog not properly configured - skipping initialization")
            return

        try:
            # Import Datadog modules
            from ddtrace import patch_all, tracer
            from ddtrace.runtime import RuntimeMetrics

            # Store tracer reference
            self._tracer = tracer

            # Patch all supported libraries
            logger.info("Initializing Datadog tracing...")
            # Enable Data Streams Monitoring for Kafka if configured
            # This is sometimes required when using manual instrumentation instead of ddtrace-run
            dsm_enabled = os.getenv('DD_DATA_STREAMS_ENABLED', 'false').lower() == 'true'
            if dsm_enabled:
                from ddtrace import config
                config.kafka["data_streams_enabled"] = True
                
                # CRITICAL: Manually import datastreams to register hooks
                # In manual instrumentation, this module is often not imported automatically
                try:
                    import ddtrace.internal.datastreams
                    logger.info("Explicitly imported ddtrace.internal.datastreams for DSM hooks")
                except ImportError:
                    logger.warning("Failed to import ddtrace.internal.datastreams - DSM may not work")

                logger.info("Explicitly enabled Data Streams Monitoring for Kafka")

            patch_all(
                logging=True,
                httpx=True,
                pymongo=True,
                psycopg=True,
                boto=True,
                openai=True,
                fastapi=True,
                kafka=True
            )
            logger.info("Datadog tracing initialized with patch_all()")

            # Enable runtime metrics
            RuntimeMetrics.enable()
            logger.info("Datadog runtime metrics enabled")

            # Conditionally enable LLM Observability
            llmobs_enabled = (
                os.getenv('DD_LLMOBS_ENABLED', 'true').lower() == 'true' and
                os.getenv('DD_LLMOBS_EVALUATORS_ENABLED', 'true').lower() == 'true'
            )
            if llmobs_enabled:
                try:
                    from ddtrace.llmobs import LLMObs
                    LLMObs.enable()
                    logger.info("Datadog LLM Observability enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable LLM Observability: {e}")

            # Conditionally enable profiler
            profiler_enabled = os.getenv('DD_PROFILING_ENABLED', 'true').lower() == 'true'
            if profiler_enabled:
                try:
                    from ddtrace.profiling import Profiler
                    prof = Profiler()
                    prof.start()
                    logger.info("Datadog profiler enabled and started")
                except Exception as e:
                    logger.warning(f"Failed to enable Datadog profiler: {e}")

            self._initialized = True
            logger.info("Datadog provider fully initialized")

        except ImportError as e:
            logger.error(f"Failed to import Datadog modules: {e}")
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Datadog: {e}")
            self._enabled = False

    def trace_decorator(self, name: str, **kwargs) -> Callable:
        """
        Return a decorator for tracing function/method execution.

        Args:
            name: Name of the trace span (used as resource if not specified)
            **kwargs: Datadog-specific options (service, resource, span_type, etc.)

        Returns:
            Decorator that wraps the function with Datadog tracing

        Example:
            @provider.trace_decorator("api.upload", service="s3-service")
            async def upload_file():
                ...
        """
        if not self._enabled or not self._tracer:
            # Return a no-op decorator if Datadog not enabled
            def noop_decorator(func: Callable) -> Callable:
                return func
            return noop_decorator

        from ddtrace import tracer

        # Use tracer.wrap() for automatic span creation
        # Set resource to name if not provided
        if 'resource' not in kwargs:
            kwargs['resource'] = name

        return tracer.wrap(name=name, **kwargs)

    @contextmanager
    def trace_context(self, name: str, **kwargs) -> ContextManager:
        """
        Return a context manager for manual trace span creation.

        Args:
            name: Name of the trace span
            **kwargs: Datadog-specific span configuration

        Yields:
            Datadog span object (or None if disabled)

        Example:
            with provider.trace_context("database.query") as span:
                result = execute_query()
                if span:
                    span.set_tag("rows", len(result))
        """
        if not self._enabled or not self._tracer:
            # Return a no-op context manager
            yield None
            return

        from ddtrace import tracer

        # Use tracer.trace() context manager
        with tracer.trace(name, **kwargs) as span:
            yield span

    def record_error(
        self,
        exception: Exception,
        error_type: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error with Datadog APM.

        Args:
            exception: The exception to record
            error_type: Classification of the error
            tags: Key-value pairs for filtering
            context: Additional context data

        Sets error tags on the current span if available.
        """
        if not self._enabled or not self._tracer:
            return

        try:
            from ddtrace import tracer
            from ddtrace.constants import ERROR_MSG, ERROR_TYPE, ERROR_STACK

            span = tracer.current_span()
            if not span:
                logger.debug("No active span to record error")
                return

            # Mark span as error
            span.error = 1

            # Set error details
            span.set_tag(ERROR_MSG, str(exception))
            if error_type:
                span.set_tag(ERROR_TYPE, error_type)
            span.set_tag(ERROR_STACK, traceback.format_exc())

            # Add service tags
            dd_service = os.getenv('DD_SERVICE', 'fastapi-app')
            dd_env = os.getenv('DD_ENV', 'dev')
            dd_version = os.getenv('DD_VERSION', '1.0')

            span.set_tag("service", dd_service)
            span.set_tag("env", dd_env)
            span.set_tag("version", dd_version)

            # Add custom tags
            if tags:
                for key, value in tags.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        # Handle tuple format: (key, value)
                        span.set_tag(key[0], key[1])
                    else:
                        span.set_tag(str(key), str(value))

            # Add context as tags
            if context:
                for key, value in context.items():
                    span.set_tag(f"context.{key}", str(value))

            logger.debug(f"Recorded error in Datadog: {error_type or 'unknown'}")

        except Exception as e:
            logger.error(f"Failed to record error in Datadog: {e}")

    def record_event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Record a custom event via Datadog Events API.

        Note: This requires DD_API_KEY and DD_APP_KEY to be configured.
        Falls back to logging if API keys not available.

        Args:
            title: Event title
            text: Event description
            alert_type: Severity (info, warning, error, success)
            tags: List of tags
            **kwargs: Additional event fields (priority, aggregation_key, etc.)
        """
        # Note: This method could call the existing Datadog Events API from src/datadog.py
        # For now, we log the event. Full implementation would require HTTP client.
        logger.info(f"Datadog Event: [{alert_type}] {title} - {text} (tags: {tags})")

        # In a full implementation, this would POST to Datadog Events API
        # Similar to the post_datadog_event function in src/datadog.py

    def set_user_context(self, user_id: str, **kwargs) -> None:
        """
        Associate user information with the current trace.

        Args:
            user_id: Unique user identifier
            **kwargs: Additional user metadata (email, username, etc.)
        """
        if not self._enabled or not self._tracer:
            return

        try:
            from ddtrace import tracer

            span = tracer.current_span()
            if span:
                span.set_tag("usr.id", user_id)
                for key, value in kwargs.items():
                    span.set_tag(f"usr.{key}", str(value))
                logger.debug(f"Set user context: {user_id}")

        except Exception as e:
            logger.error(f"Failed to set user context: {e}")

    def add_tags(self, tags: Dict[str, Any]) -> None:
        """
        Add tags to the current span.

        Args:
            tags: Key-value pairs to add as tags
        """
        if not self._enabled or not self._tracer:
            return

        try:
            from ddtrace import tracer

            span = tracer.current_span()
            if span:
                for key, value in tags.items():
                    span.set_tag(str(key), str(value))
                logger.debug(f"Added {len(tags)} tags to current span")

        except Exception as e:
            logger.error(f"Failed to add tags: {e}")

    @property
    def name(self) -> str:
        """Return provider name."""
        return "datadog"

    @property
    def is_enabled(self) -> bool:
        """Check if Datadog is enabled and initialized."""
        return self._enabled and self._initialized
