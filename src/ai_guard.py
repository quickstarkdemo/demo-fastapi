"""
Datadog AI Guard integration for real-time LLM security.

Provides:
- Reusable evaluate helper for inline LLM protection
- FastAPI endpoints for testing / playground evaluation
- Protection against prompt injection, jailbreaking, tool misuse, and PII exfiltration

Requires ddtrace >= 3.18.0 and DD_AI_GUARD_ENABLED=true.
"""

import os
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router_ai_guard = APIRouter(tags=["AI Guard"])

_client = None
_available = False


def _ensure_client():
    """Lazily create the AI Guard client on first use."""
    global _client, _available

    if _client is not None:
        return _client

    if os.getenv("DD_AI_GUARD_ENABLED", "").lower() != "true":
        logger.info("AI Guard disabled (DD_AI_GUARD_ENABLED != true)")
        _available = False
        return None

    try:
        from ddtrace.appsec.ai_guard import new_ai_guard_client
        _client = new_ai_guard_client()
        _available = True
        logger.info("AI Guard client initialised successfully")
        return _client
    except ImportError:
        logger.warning(
            "AI Guard SDK not available — upgrade to ddtrace>=3.18.0"
        )
        _available = False
        return None
    except Exception as exc:
        logger.error("Failed to create AI Guard client: %s", exc)
        _available = False
        return None


def is_ai_guard_available() -> bool:
    """Check whether AI Guard is configured and usable."""
    _ensure_client()
    return _available


def evaluate_messages(messages: list, *, block: bool = False):
    """
    Evaluate a list of messages through AI Guard.

    Args:
        messages: List of ddtrace AI Guard ``Message`` objects.
        block: If True the SDK raises ``AIGuardAbortError`` on DENY / ABORT
               when the service is configured with blocking enabled.

    Returns:
        Evaluation result with ``action`` (ALLOW / DENY / ABORT) and ``reason``,
        or None when AI Guard is unavailable.
    """
    client = _ensure_client()
    if client is None:
        return None

    from ddtrace.appsec.ai_guard import Options
    return client.evaluate(messages=messages, options=Options(block=block))


def evaluate_prompt(
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    block: bool = False,
):
    """
    Convenience wrapper: evaluate a single user prompt (with optional system prompt).

    Returns:
        Evaluation dict {"action": ..., "reason": ...} or None.
    """
    from ddtrace.appsec.ai_guard import Message

    msgs = []
    if system_prompt:
        msgs.append(Message(role="system", content=system_prompt))
    msgs.append(Message(role="user", content=user_prompt))

    result = evaluate_messages(msgs, block=block)
    if result is None:
        return None
    return {"action": str(result.action), "reason": str(result.reason)}


def evaluate_tool_call(
    tool_name: str,
    arguments: str,
    *,
    call_id: str = "call_0",
    block: bool = False,
):
    """
    Evaluate a tool call through AI Guard.

    Returns:
        Evaluation dict {"action": ..., "reason": ...} or None.
    """
    from ddtrace.appsec.ai_guard import Function, Message, ToolCall

    msgs = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    function=Function(name=tool_name, arguments=arguments),
                )
            ],
        )
    ]
    result = evaluate_messages(msgs, block=block)
    if result is None:
        return None
    return {"action": str(result.action), "reason": str(result.reason)}


# ---------------------------------------------------------------------------
# Pydantic request / response models for the REST endpoints
# ---------------------------------------------------------------------------

class AIGuardMessageModel(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant, or tool")
    content: Optional[str] = Field(None, description="Text content of the message")
    tool_calls: Optional[List[dict]] = Field(
        None,
        description="Tool calls (for assistant messages)",
    )
    tool_call_id: Optional[str] = Field(
        None,
        description="ID of the tool call this message responds to (for tool messages)",
    )


class AIGuardEvaluateRequest(BaseModel):
    messages: List[AIGuardMessageModel] = Field(
        ..., min_length=1, description="Conversation messages to evaluate"
    )
    block: bool = Field(
        False,
        description="Raise on DENY/ABORT when blocking is enabled for the service",
    )


class AIGuardPromptRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to evaluate")
    system_prompt: Optional[str] = Field(
        None, description="Optional system prompt for context"
    )
    block: bool = False


class AIGuardToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool being called")
    arguments: str = Field(..., description="JSON-serialised arguments for the tool")
    block: bool = False


class AIGuardEvaluateResponse(BaseModel):
    action: str = Field(..., description="ALLOW, DENY, or ABORT")
    reason: str = Field(..., description="Human-readable rationale")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router_ai_guard.get("/ai-guard/status")
async def ai_guard_status():
    """Check whether AI Guard is available and configured."""
    available = is_ai_guard_available()
    return {
        "available": available,
        "enabled": os.getenv("DD_AI_GUARD_ENABLED", "false"),
        "ddtrace_min_version": "3.18.0",
    }


@router_ai_guard.post(
    "/ai-guard/evaluate",
    response_model=AIGuardEvaluateResponse,
)
async def ai_guard_evaluate(request: AIGuardEvaluateRequest):
    """
    Evaluate a full conversation through Datadog AI Guard.

    Accepts the standard message format (system / user / assistant / tool)
    and returns an action (ALLOW / DENY / ABORT) with a reason.
    """
    if not is_ai_guard_available():
        raise HTTPException(
            status_code=503,
            detail="AI Guard is not available. Ensure DD_AI_GUARD_ENABLED=true and ddtrace>=3.18.0.",
        )

    from ddtrace.appsec.ai_guard import Function, Message, ToolCall

    msgs = []
    for m in request.messages:
        if m.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", "call_0"),
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in m.tool_calls
            ]
            msgs.append(Message(role=m.role, tool_calls=tool_calls))
        elif m.tool_call_id:
            msgs.append(
                Message(role=m.role, content=m.content or "", tool_call_id=m.tool_call_id)
            )
        else:
            msgs.append(Message(role=m.role, content=m.content or ""))

    try:
        result = evaluate_messages(msgs, block=request.block)
    except Exception as exc:
        logger.error("AI Guard evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"AI Guard evaluation error: {exc}")

    if result is None:
        raise HTTPException(status_code=503, detail="AI Guard returned no result")

    return AIGuardEvaluateResponse(
        action=str(result.action),
        reason=str(result.reason),
    )


@router_ai_guard.post(
    "/ai-guard/evaluate-prompt",
    response_model=AIGuardEvaluateResponse,
)
async def ai_guard_evaluate_prompt(request: AIGuardPromptRequest):
    """
    Quick evaluation of a single user prompt (with optional system prompt).
    """
    if not is_ai_guard_available():
        raise HTTPException(
            status_code=503,
            detail="AI Guard is not available. Ensure DD_AI_GUARD_ENABLED=true and ddtrace>=3.18.0.",
        )

    try:
        result = evaluate_prompt(
            request.prompt,
            system_prompt=request.system_prompt,
            block=request.block,
        )
    except Exception as exc:
        logger.error("AI Guard prompt evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"AI Guard evaluation error: {exc}")

    if result is None:
        raise HTTPException(status_code=503, detail="AI Guard returned no result")

    return AIGuardEvaluateResponse(**result)


@router_ai_guard.post(
    "/ai-guard/evaluate-tool-call",
    response_model=AIGuardEvaluateResponse,
)
async def ai_guard_evaluate_tool_call(request: AIGuardToolCallRequest):
    """
    Evaluate a tool call through AI Guard before executing it.
    """
    if not is_ai_guard_available():
        raise HTTPException(
            status_code=503,
            detail="AI Guard is not available. Ensure DD_AI_GUARD_ENABLED=true and ddtrace>=3.18.0.",
        )

    try:
        result = evaluate_tool_call(
            request.tool_name,
            request.arguments,
            block=request.block,
        )
    except Exception as exc:
        logger.error("AI Guard tool-call evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"AI Guard evaluation error: {exc}")

    if result is None:
        raise HTTPException(status_code=503, detail="AI Guard returned no result")

    return AIGuardEvaluateResponse(**result)
