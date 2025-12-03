"""
Gemini image generation endpoints.

Uses the Gemini image generation endpoint (Imagen 3.5 Flash by default).
"""

import os
import logging
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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


def _extract_image_payload(data: dict) -> Tuple[str, str]:
    """
    Extract base64 image payload and mime type from Gemini image response.
    Supports both inlineData (camelCase) and inline_data (legacy) plus a minimal images array fallback.
    """
    try:
        candidates = data.get("candidates", [])
        parts = candidates[0]["content"]["parts"]
        inline_data = None
        for p in parts:
            if "inlineData" in p:
                inline_data = p.get("inlineData")
                break
            if "inline_data" in p:
                inline_data = p.get("inline_data")
                break

        if inline_data and "data" in inline_data:
            mime = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
            return inline_data.get("data"), mime
    except Exception as e:
        logger.debug("Inline data parse failed: %s", e)

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
            "imageConfig": {
                "aspectRatio": f"{width}:{height}",
            },
        },
        "safetySettings": [],
    }
    return url, payload


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
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Gemini API error ({resp.status_code}): {text[:400]}",
                )
            data = resp.json()

        image_b64, mime_type = _extract_image_payload(data)
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
