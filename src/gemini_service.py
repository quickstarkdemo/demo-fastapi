"""
Gemini image generation endpoints.

Uses the Gemini image generation endpoint (Imagen 3.5 Flash by default).
"""

import os
import logging
import sys
from contextlib import contextmanager
from typing import List, Optional, Tuple
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


class GeminiImageResponse(BaseModel):
    model: str
    mime_type: str
    image_base64: str
    prompt: str
    size: str
    thought: Optional[bool] = None
    thought_signature: Optional[str] = None


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


@contextmanager
def _llm_span(operation: str, model_name: str, prompt: str, extra_tags: Optional[dict] = None):
    """
    Best-effort Datadog span for LLM requests so Gemini traffic shows up in LLM monitoring.
    """
    if not tracer:
        yield None
        return

    with tracer.trace(
        "llm.request",
        service=os.getenv("DD_SERVICE", "fastapi-app"),
        resource=operation,
        span_type="llm",
    ) as span:
        try:
            span.set_tag("span.kind", "llm")
            span.set_tag("component", "gemini")
            span.set_tag("llm.provider", "google")
            span.set_tag("llm.model", model_name)
            span.set_tag("llm.operation", operation)
            if prompt:
                span.set_tag("llm.request.prompt", prompt)
                span.set_metric("llm.request.prompt_length", len(prompt))
            if extra_tags:
                for key, value in extra_tags.items():
                    span.set_tag(key, str(value))
            yield span
        except Exception:
            span.set_exc_info(*sys.exc_info())
            raise


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
    Returns a dict with keys: data, mime_type, thought, thought_signature (optional).
    """
    result = {"data": None, "mime_type": "image/png"}
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            return result
            
        parts = candidates[0].get("content", {}).get("parts", [])
        
        # Look for thought/signature in any part
        for p in parts:
            if p.get("thought"):
                result["thought"] = True
            if "thought_signature" in p:
                result["thought_signature"] = p["thought_signature"]
                
            # Look for image data
            if "inlineData" in p:
                inline = p["inlineData"]
                if "data" in inline:
                    result["data"] = inline["data"]
                    result["mime_type"] = inline.get("mimeType", "image/png")
            elif "inline_data" in p:
                inline = p["inline_data"]
                if "data" in inline:
                    result["data"] = inline["data"]
                    result["mime_type"] = inline.get("mimeType", "image/png")
                    
        return result
            
    except Exception as e:
        logger.debug("Inline data parse failed: %s", e)
        return result

    # Fallback shape some beta responses use
    images = data.get("images") or data.get("image")
    if images:
        first = images[0] if isinstance(images, list) else images
        encoded = first.get("image") or first.get("data")
        mime = first.get("mime_type") or "image/png"
        if encoded:
            return encoded, mime

    raise ValueError("Could not parse image payload from Gemini response")


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
            return {"text": part.text}
        part_dict = {
            "inlineData": {
                "mimeType": part.mime_type or "image/png",
                "data": part.image_base64,
            }
        }
        if part.thought:
            part_dict["thought"] = True
        if part.thought_signature:
            part_dict["thought_signature"] = part.thought_signature
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
        ) as span:
            async with httpx.AsyncClient(timeout=60) as client:
                # Use header as in Google’s examples; keep query param for compatibility
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

            return GeminiImageResponse(
                model=model_name,
                mime_type=mime_type,
                image_base64=image_b64,
                prompt=request.prompt,
                size=f"{width}x{height}",
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
            },
        ) as span:
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

            return GeminiImageResponse(
                model=model_name,
                mime_type=mime_type,
                image_base64=image_b64,
                prompt=prompt_summary,
                size=size_label,
                thought=extracted.get("thought"),
                thought_signature=extracted.get("thought_signature"),
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Gemini multi-turn editing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini image editing failed")
