"""
Gemini image generation endpoints.

Uses the Gemini image generation endpoint (Imagen 3.5 Flash by default).
"""

import os
import logging
import sys
from contextlib import contextmanager
from typing import List, Optional, Tuple, Any
import math

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Optional Datadog tracer (fallback to no-op if ddtrace is missing)
try:
    from ddtrace import tracer
except Exception:
    tracer = None

# Optional LLMObs for proper LLM span tracking
_llmobs_available = False
_LLMObs = None
try:
    from ddtrace.llmobs import LLMObs as _LLMObs
    _llmobs_available = True
except Exception:
    pass

router_gemini = APIRouter(tags=["Gemini"])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Align default to the doc’s primary model (Nano Banana): gemini-2.5-flash-image
DEFAULT_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


class GeminiImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate the image")
    size: Optional[str] = Field(
        default="1024x1024",
        description="Image size as WIDTHxHEIGHT (e.g., 512x512, 1024x1024)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Gemini image model name (defaults to imagen-3.5-flash)",
    )


class GeminiPartSignature(BaseModel):
    """Metadata for a response part, including thought signatures for multi-turn."""
    index: int = Field(..., description="Part index in the response")
    thought: Optional[bool] = Field(default=None, description="Whether this part is a thought")
    thought_signature: Optional[str] = Field(default=None, description="Signature for this part")
    mime_type: Optional[str] = Field(default=None, description="MIME type if this is image data")
    has_image: Optional[bool] = Field(default=False, description="Whether this part contains image data")


class GeminiImageResponse(BaseModel):
    model: str
    mime_type: str
    image_base64: str
    prompt: str
    size: str
    thought: Optional[bool] = None
    thought_signature: Optional[str] = None
    # All part signatures for multi-turn - client MUST pass these back
    response_parts: Optional[List[GeminiPartSignature]] = Field(
        default=None,
        description="All response parts with signatures. Pass these back in multi-turn for proper context."
    )


class GeminiImagePart(BaseModel):
    """
    A single part of a Gemini message: either text or an inline base64 image.
    """

    text: Optional[str] = Field(
        default=None, description="Plain text content for this part"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image to send back for editing",
    )
    mime_type: Optional[str] = Field(
        default="image/png",
        description="Image MIME type for image_base64 parts",
    )
    thought: Optional[bool] = Field(
        default=None,
        description="Whether this part represents a thought (Chain of Thought)",
    )
    thought_signature: Optional[str] = Field(
        default=None,
        description="Signature for thought parts, required for multi-turn CoT",
    )

    @model_validator(mode="after")
    def validate_content(self):
        if bool(self.text) == bool(self.image_base64):
            raise ValueError("Provide exactly one of text or image_base64 in a part")
        return self


class GeminiImageMessage(BaseModel):
    role: str = Field(
        ...,
        description="Message role ('user' or 'assistant')",
    )
    parts: List[GeminiImagePart]

    @field_validator("role")
    def normalize_role(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"user", "assistant", "model"}:
            raise ValueError("role must be 'user', 'assistant', or 'model'")
        return normalized

    @model_validator(mode="after")
    def ensure_parts(self):
        if not (self.parts or []):
            raise ValueError("messages must include at least one part")
        return self


class GeminiMultiTurnRequest(BaseModel):
    messages: List[GeminiImageMessage] = Field(
        ...,
        description="Ordered conversation history to send to Gemini",
    )
    size: Optional[str] = Field(
        default=None,
        description="Optional WIDTHxHEIGHT hint; sets aspectRatio if valid",
    )
    model: Optional[str] = Field(
        default=None,
        description="Gemini image model name (defaults to gemini-2.5-flash-image)",
    )

    @model_validator(mode="after")
    def validate_messages(self):
        if not (self.messages or []):
            raise ValueError("messages cannot be empty for multi-turn editing")
        return self


def _is_llmobs_enabled() -> bool:
    """Check if LLMObs is available and enabled."""
    if not _llmobs_available or not _LLMObs:
        return False
    try:
        return _LLMObs._instance is not None
    except Exception:
        return False


@contextmanager
def _llm_span(operation: str, model_name: str, prompt: str, extra_tags: Optional[dict] = None):
    """
    Datadog LLMObs-compatible span for LLM requests.

    Uses LLMObs.llm() when available for proper LLM Observability integration,
    falls back to regular tracer spans otherwise.
    """
    span_context = {"span": None, "llmobs_enabled": False}

    if _is_llmobs_enabled():
        # Use LLMObs for proper span kind registration
        try:
            with _LLMObs.llm(
                model_name=model_name,
                name=operation,
                model_provider="google"
            ) as span:
                span_context["span"] = span
                span_context["llmobs_enabled"] = True
                if extra_tags:
                    for key, value in extra_tags.items():
                        span.set_tag(key, str(value))
                yield span_context
        except Exception as e:
            logger.warning("LLMObs span creation failed, falling back to tracer: %s", e)
            # Fall through to tracer fallback
            span_context["llmobs_enabled"] = False
            if tracer:
                with tracer.trace(
                    "llm.request",
                    service=os.getenv("DD_SERVICE", "fastapi-app"),
                    resource=operation,
                    span_type="llm",
                ) as span:
                    span.set_tag("llm.provider", "google")
                    span.set_tag("llm.model", model_name)
                    span.set_tag("llm.operation", operation)
                    if prompt:
                        span.set_tag("llm.request.prompt", prompt[:500])
                    if extra_tags:
                        for key, value in extra_tags.items():
                            span.set_tag(key, str(value))
                    span_context["span"] = span
                    yield span_context
            else:
                yield span_context
    elif tracer:
        # Fallback to regular tracer when LLMObs not enabled
        with tracer.trace(
            "llm.request",
            service=os.getenv("DD_SERVICE", "fastapi-app"),
            resource=operation,
            span_type="llm",
        ) as span:
            span.set_tag("llm.provider", "google")
            span.set_tag("llm.model", model_name)
            span.set_tag("llm.operation", operation)
            if prompt:
                span.set_tag("llm.request.prompt", prompt[:500])
            if extra_tags:
                for key, value in extra_tags.items():
                    span.set_tag(key, str(value))
            span_context["span"] = span
            yield span_context
    else:
        yield span_context


def _annotate_llm_span(span_context: dict, input_data: Any = None, output_data: Any = None, metadata: dict = None):
    """
    Annotate an LLMObs span with input/output data.
    Only works when LLMObs is enabled.
    """
    if not span_context.get("llmobs_enabled") or not _LLMObs:
        return

    try:
        span = span_context.get("span")
        if span:
            _LLMObs.annotate(
                span=span,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata
            )
    except Exception as e:
        logger.debug("Failed to annotate LLMObs span: %s", e)


def _parse_size(size: str) -> Tuple[int, int]:
    """
    Convert WIDTHxHEIGHT into integers. Falls back to 1024x1024 on error.
    """
    try:
        width_str, height_str = size.lower().split("x")
        width, height = int(width_str), int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        return width, height
    except Exception:
        logger.warning("Invalid size '%s', defaulting to 1024x1024", size)
        return 1024, 1024


def _normalize_aspect_ratio(width: int, height: int) -> Optional[str]:
    """
    Normalize width/height to an allowed aspect ratio string if possible.
    """
    allowed = {
        "1:1",
        "2:3", "3:2",
        "3:4", "4:3",
        "4:5", "5:4",
        "9:16", "16:9",
        "21:9",
    }
    try:
        g = math.gcd(width, height)
        ratio = f"{width // g}:{height // g}"
        if ratio in allowed:
            return ratio
    except Exception as e:
        logger.debug("Aspect ratio normalize failed: %s", e)
    return None


def _extract_image_payload(data: dict) -> dict:
    """
    Extract base64 image payload, mime type, and CoT metadata from Gemini response.

    Returns a dict with keys:
        - data: base64 image data
        - mime_type: image MIME type
        - thought: whether any thought part was present
        - thought_signature: signature from the image part (for backward compat)
        - response_parts: list of all parts with their signatures (for multi-turn)

    IMPORTANT: For multi-turn conversations, the client MUST pass back ALL
    response_parts with their signatures to avoid 400 errors.
    """
    result = {
        "data": None,
        "mime_type": "image/png",
        "thought": None,
        "thought_signature": None,
        "response_parts": []
    }

    try:
        candidates = data.get("candidates", [])
        if not candidates:
            return result

        parts = candidates[0].get("content", {}).get("parts", [])

        # Extract ALL parts with their signatures for multi-turn support
        for idx, p in enumerate(parts):
            part_info = {
                "index": idx,
                "thought": p.get("thought"),
                "thought_signature": p.get("thought_signature"),
                "mime_type": None,
                "has_image": False
            }

            # Track if any part has thought
            if p.get("thought"):
                result["thought"] = True

            # Look for image data in this part
            inline = p.get("inlineData") or p.get("inline_data")
            if inline:
                mime_type = inline.get("mimeType") or inline.get("mime_type", "image/png")
                part_info["mime_type"] = mime_type

                if "data" in inline:
                    part_info["has_image"] = True
                    # Store the first image found as the primary result
                    if result["data"] is None:
                        result["data"] = inline["data"]
                        result["mime_type"] = mime_type
                        # Store this signature for backward compatibility
                        if p.get("thought_signature"):
                            result["thought_signature"] = p["thought_signature"]

            result["response_parts"].append(part_info)

        return result

    except Exception as e:
        logger.debug("Inline data parse failed: %s", e)
        return result


def _build_payload(model_name: str, prompt: str, width: int, height: int) -> Tuple[str, dict]:
    """
    Build the appropriate endpoint URL and payload based on the model.
    - Imagen models use the imagegeneration:generate endpoint.
    - Other Gemini models (e.g., gemini-3-pro-image-preview) use generateContent.
    """
    # Imagen family uses the dedicated imagegeneration endpoint
    if model_name.startswith("imagen-"):
        url = "https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:generate"
        payload = {
            "model": model_name,
            "prompt": {"text": prompt},
            "imageGenerationConfig": {
                "sampleCount": 1,
                "height": height,
                "width": width,
            },
        }
        return url, payload

    # Gemini v1/v2/v3 image-capable models use generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    aspect_ratio = _normalize_aspect_ratio(width, height)
    image_config = {}
    if aspect_ratio:
        image_config["aspectRatio"] = aspect_ratio

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ],
            }
        ],
        # Hint that we want an image response
        "generationConfig": {
            # Match doc guidance: explicitly request image
            "responseModalities": ["IMAGE"],
        },
        "safetySettings": [],
    }
    if image_config:
        payload["generationConfig"]["imageConfig"] = image_config

    return url, payload


def _build_multi_turn_payload(
    model_name: str,
    messages: List[GeminiImageMessage],
    size: Optional[str] = None,
) -> Tuple[str, dict, str]:
    """
    Build payload for multi-turn image editing via generateContent.
    Returns (url, payload, size_label).
    """
    if model_name.startswith("imagen-"):
        raise HTTPException(
            status_code=400,
            detail="Multi-turn editing requires a Gemini image model (gemini-*)",
        )

    width = height = None
    size_label = "unspecified"
    image_config = {}
    if size:
        width, height = _parse_size(size)
        size_label = f"{width}x{height}"
        aspect_ratio = _normalize_aspect_ratio(width, height)
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio

    def _to_gemini_part(part: GeminiImagePart) -> dict:
        if part.text is not None:
            text_part = {"text": part.text}
            # Text parts may also need signatures in multi-turn
            if part.thought:
                text_part["thought"] = True
            if part.thought_signature:
                text_part["thought_signature"] = part.thought_signature
            return text_part

        # Image part - MUST include thought_signature for multi-turn
        part_dict = {
            "inlineData": {
                "mimeType": part.mime_type or "image/png",
                "data": part.image_base64,
            }
        }
        if part.thought:
            part_dict["thought"] = True

        # Use provided signature, or skip validator if not provided
        # This allows multi-turn to work even without proper signature passthrough
        # See: https://ai.google.dev/gemini-api/docs/thought-signatures
        if part.thought_signature:
            part_dict["thought_signature"] = part.thought_signature
        else:
            # Use skip validator to bypass signature validation for client simplicity
            part_dict["thought_signature"] = "skip_thought_signature_validator"

        return part_dict

    contents = []
    for msg in messages:
        parts = [_to_gemini_part(p) for p in msg.parts]
        # Map 'assistant' to 'model' for the API
        role = "model" if msg.role == "assistant" else msg.role
        contents.append({"role": role, "parts": parts})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload: dict = {
        "contents": contents,
        "generationConfig": {"responseModalities": ["IMAGE"]},
        "safetySettings": [],
    }
    if image_config:
        payload["generationConfig"]["imageConfig"] = image_config

    return url, payload, size_label


def _last_user_prompt(messages: List[GeminiImageMessage]) -> str:
    """
    Best-effort extraction of the latest user text prompt for response metadata.
    """
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        for part in reversed(msg.parts):
            if part.text:
                return part.text
    return ""


@router_gemini.post("/gemini-generate-image", response_model=GeminiImageResponse)
async def generate_gemini_image(request: GeminiImageRequest):
    """
    Generate an image with Gemini (Imagen 3.5 Flash) and return base64 content.

    The response includes `response_parts` with thought signatures that MUST be
    passed back in subsequent multi-turn requests to avoid validation errors.
    """
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    model_name = request.model or DEFAULT_MODEL
    width, height = _parse_size(request.size or "1024x1024")

    url, payload = _build_payload(model_name, request.prompt, width, height)

    try:
        with _llm_span(
            operation="gemini.generate_image",
            model_name=model_name,
            prompt=request.prompt,
            extra_tags={
                "llm.request.size": f"{width}x{height}",
                "llm.request.endpoint": url,
            },
        ) as span_context:
            span = span_context.get("span") if span_context else None

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"x-goog-api-key": api_key},
                    json=payload,
                )
                if resp.status_code >= 400:
                    text = resp.text
                    logger.error("Gemini API error %s: %s", resp.status_code, text)
                    if span:
                        span.set_tag("error", True)
                        span.set_tag("llm.response.body", text[:400])
                        span.set_tag("http.status_code", str(resp.status_code))
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Gemini API error ({resp.status_code}): {text[:400]}",
                    )
                data = resp.json()

            extracted = _extract_image_payload(data)
            image_b64 = extracted.get("data")
            mime_type = extracted.get("mime_type")

            if span:
                span.set_tag("llm.response.mime_type", mime_type)
                span.set_tag("llm.response.size", f"{width}x{height}")
                span.set_tag("llm.response.model", model_name)
                span.set_tag("http.status_code", str(resp.status_code))

            # Annotate LLMObs span with input/output for proper tracking
            _annotate_llm_span(
                span_context,
                input_data=request.prompt,
                output_data={"image_generated": bool(image_b64), "mime_type": mime_type},
                metadata={"size": f"{width}x{height}", "model": model_name}
            )

            # Build response_parts for multi-turn support
            response_parts = None
            if extracted.get("response_parts"):
                response_parts = [
                    GeminiPartSignature(
                        index=p["index"],
                        thought=p.get("thought"),
                        thought_signature=p.get("thought_signature"),
                        mime_type=p.get("mime_type"),
                        has_image=p.get("has_image", False)
                    )
                    for p in extracted["response_parts"]
                ]

            return GeminiImageResponse(
                model=model_name,
                mime_type=mime_type,
                image_base64=image_b64,
                prompt=request.prompt,
                size=f"{width}x{height}",
                thought=extracted.get("thought"),
                thought_signature=extracted.get("thought_signature"),
                response_parts=response_parts,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Gemini image generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini image generation failed")


@router_gemini.post("/gemini-edit-image", response_model=GeminiImageResponse)
async def edit_gemini_image(request: GeminiMultiTurnRequest):
    """
    Perform multi-turn image editing by sending prior conversation (text + images) to Gemini.

    IMPORTANT: For multi-turn to work, you should either:
    1. Pass thought_signature from previous responses in the image parts, OR
    2. The API will automatically use skip_thought_signature_validator for convenience.

    The response includes `response_parts` for subsequent turns.
    """
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    model_name = request.model or DEFAULT_MODEL
    try:
        url, payload, size_label = _build_multi_turn_payload(
            model_name=model_name,
            messages=request.messages,
            size=request.size,
        )

        prompt_summary = _last_user_prompt(request.messages)
        with _llm_span(
            operation="gemini.edit_image",
            model_name=model_name,
            prompt=prompt_summary,
            extra_tags={
                "llm.request.size": size_label,
                "llm.request.endpoint": url,
                "llm.request.turn_count": len(request.messages),
            },
        ) as span_context:
            span = span_context.get("span") if span_context else None

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"x-goog-api-key": api_key},
                    json=payload,
                )
                if resp.status_code >= 400:
                    text = resp.text
                    logger.error("Gemini API error %s: %s", resp.status_code, text)
                    if span:
                        span.set_tag("error", True)
                        span.set_tag("llm.response.body", text[:400])
                        span.set_tag("http.status_code", str(resp.status_code))
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Gemini API error ({resp.status_code}): {text[:400]}",
                    )
                data = resp.json()

            extracted = _extract_image_payload(data)
            image_b64 = extracted.get("data")
            mime_type = extracted.get("mime_type")

            if span:
                span.set_tag("llm.response.mime_type", mime_type)
                span.set_tag("llm.response.size", size_label)
                span.set_tag("llm.response.model", model_name)
                span.set_tag("http.status_code", str(resp.status_code))

            # Annotate LLMObs span with input/output for proper tracking
            _annotate_llm_span(
                span_context,
                input_data=prompt_summary,
                output_data={"image_generated": bool(image_b64), "mime_type": mime_type},
                metadata={
                    "size": size_label,
                    "model": model_name,
                    "turn_count": len(request.messages)
                }
            )

            # Build response_parts for subsequent multi-turn support
            response_parts = None
            if extracted.get("response_parts"):
                response_parts = [
                    GeminiPartSignature(
                        index=p["index"],
                        thought=p.get("thought"),
                        thought_signature=p.get("thought_signature"),
                        mime_type=p.get("mime_type"),
                        has_image=p.get("has_image", False)
                    )
                    for p in extracted["response_parts"]
                ]

            return GeminiImageResponse(
                model=model_name,
                mime_type=mime_type,
                image_base64=image_b64,
                prompt=prompt_summary,
                size=size_label,
                thought=extracted.get("thought"),
                thought_signature=extracted.get("thought_signature"),
                response_parts=response_parts,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Gemini multi-turn editing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini image editing failed")
