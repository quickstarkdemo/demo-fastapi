# Gemini LLM Observability Enrichment Plan

**Date**: 2026-01-23
**Issue**: Sparse Datadog LLM Observability spans for Gemini image generation
**Goal**: Rich hierarchical spans like OpenAI chatbot app

## Current vs Target Comparison

### Current State (Sparse)
```
gemini.edit_image (single llm span - 12.5s)
  [no child spans]
  Input: "Generate a Blue Morpho Butterfly"
  Metadata: {model, size, turn_count}
  Tokens: 0 (N/A for image gen)
  Evaluations: 0
```

### Target State (Rich - Like OpenAI App)
```
image_generation_workflow (workflow span - 12.5s)
├── validate_request (tool - 0.1s)
│   ├── Input: {prompt, size, model}
│   └── Output: {validated: true, dimensions: "1024x1024"}
├── build_payload (tool - 0.2s)
│   ├── Input: {prompt, width, height, model}
│   ├── Output: {endpoint, config, aspect_ratio}
│   └── Metadata: {endpoint_type, modalities, safety_settings}
├── gemini.generate_image (llm - 11.8s)
│   ├── Input: "Generate a Blue Morpho Butterfly"
│   ├── Output: {image_bytes: 125000, mime_type: "image/png"}
│   └── Metadata: {request_bytes, response_time_ms, http_status}
└── extract_response (tool - 0.4s)
    ├── Input: {candidates_count: 1, parts_count: 2}
    ├── Output: {image_extracted: true, has_thought: true}
    └── Metadata: {thought_signature, response_parts_count}
```

## Why Current Implementation Is Sparse

### 1. **No Span Hierarchy** 🔴
**Problem**: Using single `_llm_span()` wrapper for entire operation
**Location**: `src/gemini_service.py:512-520`

```python
# Current: Everything in one span
with _llm_span("gemini.generate_image", ...) as span_context:
    url, payload = _build_payload(...)  # Not a separate span
    resp = await client.post(...)  # Not a separate span
    extracted = _extract_image_payload(data)  # Not a separate span
```

**Why This Causes Sparseness**:
- No breakdown of operation stages
- Can't see which step is slow (payload build vs API call vs extraction)
- No individual timing metrics for sub-operations
- Datadog shows single flat span instead of tree

### 2. **Missing Workflow Context** 🔴
**Problem**: Not using `LLMObs.workflow()` to establish operation hierarchy
**Impact**: Datadog doesn't recognize this as a multi-step workflow

**OpenAI App Uses**:
```python
with LLMObs.agent(name="shopist-chat"):  # Top-level agent
    with LLMObs.workflow(name="plan"):  # Workflow stage
        with LLMObs.tool(name="encode_context"):  # Individual tool
            # Operation
```

### 3. **No Tool Spans** 🔴
**Problem**: Helper functions not instrumented as tools
**Missing Tool Spans**:
- `_build_payload()` - payload construction logic
- `_parse_size()` - size validation
- `_normalize_aspect_ratio()` - aspect ratio calculation
- `_extract_image_payload()` - response parsing

### 4. **Minimal Metadata** 🟡
**Current Metadata** (line 558):
```python
metadata={"size": f"{width}x{height}", "model": model_name}
```

**Should Include**:
```python
metadata={
    # Request context
    "size": "1024x1024",
    "model": "gemini-3-pro-image-preview",
    "aspect_ratio": "1:1",
    "endpoint_type": "generateContent",

    # Configuration
    "response_modalities": ["IMAGE"],
    "safety_settings_count": 0,
    "generation_config": {...},

    # Performance
    "request_bytes": 450,
    "response_bytes": 125000,
    "api_latency_ms": 11800,
    "extraction_time_ms": 350,

    # Features
    "thought_enabled": True,
    "multi_turn": True,
    "turn_count": 1,
    "has_previous_image": False
}
```

### 5. **No Output Capture** 🟡
**Current Output** (line 557):
```python
output_data={"image_generated": bool(image_b64), "mime_type": mime_type}
```

**Should Include**:
```python
output_data={
    "image_generated": True,
    "mime_type": "image/png",
    "image_bytes": 125000,
    "image_dimensions": "1024x1024",
    "thought_present": True,
    "thought_signature": "abc123...",
    "response_parts_count": 2,
    "candidates_count": 1,
    "has_safety_flags": False
}
```

## Implementation Plan

### Step 1: Add Workflow Wrapper Function

**Create**: `_image_generation_workflow()` helper

```python
async def _image_generation_workflow(
    operation: str,
    model_name: str,
    request_data: dict,
    api_call_func: callable
) -> dict:
    """
    Workflow wrapper for image generation with hierarchical spans.

    Creates structure:
    - workflow (top level)
      - validate_request (tool)
      - build_payload (tool)
      - gemini.api_call (llm)
      - extract_response (tool)
    """
    if not _is_llmobs_enabled():
        # Fallback to flat span if LLMObs not available
        return await api_call_func()

    with _LLMObs.workflow(name=operation) as workflow_span:
        # Workflow-level annotations
        _LLMObs.annotate(
            span=workflow_span,
            input_data=request_data,
            metadata={
                "model": model_name,
                "operation_type": operation
            }
        )

        # Execute API call through child spans
        result = await api_call_func()

        # Workflow-level output
        _LLMObs.annotate(
            span=workflow_span,
            output_data=result
        )

        return result
```

### Step 2: Instrument _build_payload as Tool

**Modify**: `_build_payload()` function (line 357)

```python
def _build_payload(model_name: str, prompt: str, width: int, height: int) -> Tuple[str, dict]:
    """Build payload with LLMObs tool span."""

    if _is_llmobs_enabled():
        with _LLMObs.tool(name="build_payload") as tool_span:
            # Input tracking
            _LLMObs.annotate(
                span=tool_span,
                input_data={
                    "model": model_name,
                    "prompt": prompt[:100],
                    "dimensions": f"{width}x{height}"
                }
            )

            # Build logic (existing code)
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
                endpoint_type = "imagen"
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                aspect_ratio = _normalize_aspect_ratio(width, height)
                image_config = {}
                if aspect_ratio:
                    image_config["aspectRatio"] = aspect_ratio

                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                    },
                    "safetySettings": [],
                }
                if image_config:
                    payload["generationConfig"]["imageConfig"] = image_config

                endpoint_type = "generateContent"

            # Output tracking
            _LLMObs.annotate(
                span=tool_span,
                output_data={
                    "endpoint": url,
                    "endpoint_type": endpoint_type,
                    "payload_size_bytes": len(json.dumps(payload))
                },
                metadata={
                    "aspect_ratio": _normalize_aspect_ratio(width, height),
                    "response_modalities": payload.get("generationConfig", {}).get("responseModalities", []),
                    "safety_settings_count": len(payload.get("safetySettings", []))
                }
            )

            return url, payload
    else:
        # Fallback: existing code without instrumentation
        # [existing _build_payload logic]
        pass
```

### Step 3: Instrument API Call as LLM Span

**Refactor**: Lines 523-541 in `generate_gemini_image()`

```python
# Inside workflow context
import time

# Track API timing
api_start = time.time()

with _LLMObs.llm(
    model_name=model_name,
    name="gemini.api_call",
    model_provider="google"
) as llm_span:
    # Input annotation
    _LLMObs.annotate(
        span=llm_span,
        input_data=request.prompt,
        metadata={
            "endpoint": url,
            "request_size_bytes": len(json.dumps(payload)),
            "timeout": 60
        }
    )

    # API call
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            params={"key": api_key},
            headers={"x-goog-api-key": api_key},
            json=payload,
        )

        api_duration_ms = (time.time() - api_start) * 1000

        # Error handling
        if resp.status_code >= 400:
            text = resp.text
            logger.error("Gemini API error %s: %s", resp.status_code, text)

            # Rich error annotation
            _LLMObs.annotate(
                span=llm_span,
                output_data={"error": True, "status_code": resp.status_code},
                metadata={
                    "error_message": text[:500],
                    "http_status_code": resp.status_code,
                    "api_latency_ms": api_duration_ms
                }
            )

            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Gemini API error ({resp.status_code}): {text[:400]}",
            )

        data = resp.json()

        # Success annotation
        _LLMObs.annotate(
            span=llm_span,
            output_data={
                "success": True,
                "candidates_count": len(data.get("candidates", [])),
                "response_size_bytes": len(resp.text)
            },
            metadata={
                "http_status_code": resp.status_code,
                "api_latency_ms": api_duration_ms,
                "response_headers": dict(resp.headers)
            }
        )
```

### Step 4: Instrument _extract_image_payload as Tool

**Modify**: `_extract_image_payload()` function (line 289)

```python
def _extract_image_payload(data: dict) -> dict:
    """Extract image with LLMObs tool span."""

    if _is_llmobs_enabled():
        with _LLMObs.tool(name="extract_image_payload") as tool_span:
            extraction_start = time.time()

            # Input tracking
            candidates = data.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []

            _LLMObs.annotate(
                span=tool_span,
                input_data={
                    "candidates_count": len(candidates),
                    "parts_count": len(parts),
                    "raw_response_size": len(str(data))
                }
            )

            # Extraction logic (existing code)
            result = {
                "data": None,
                "mime_type": "image/png",
                "thought": None,
                "thought_signature": None,
                "response_parts": []
            }

            try:
                for idx, p in enumerate(parts):
                    part_info = {
                        "index": idx,
                        "thought": p.get("thought"),
                        "thought_signature": p.get("thought_signature"),
                        "mime_type": None,
                        "has_image": False
                    }

                    if p.get("thought"):
                        result["thought"] = True

                    inline = p.get("inlineData") or p.get("inline_data")
                    if inline:
                        mime_type = inline.get("mimeType") or inline.get("mime_type", "image/png")
                        part_info["mime_type"] = mime_type

                        if "data" in inline:
                            part_info["has_image"] = True
                            if result["data"] is None:
                                result["data"] = inline["data"]
                                result["mime_type"] = mime_type
                                if p.get("thought_signature"):
                                    result["thought_signature"] = p["thought_signature"]

                    result["response_parts"].append(part_info)

                extraction_duration_ms = (time.time() - extraction_start) * 1000

                # Output tracking
                _LLMObs.annotate(
                    span=tool_span,
                    output_data={
                        "image_extracted": bool(result["data"]),
                        "image_bytes": len(result["data"]) if result["data"] else 0,
                        "mime_type": result["mime_type"],
                        "thought_present": bool(result["thought"]),
                        "parts_extracted": len(result["response_parts"])
                    },
                    metadata={
                        "extraction_time_ms": extraction_duration_ms,
                        "thought_signature": result.get("thought_signature"),
                        "has_multiple_parts": len(result["response_parts"]) > 1
                    }
                )

                return result

            except Exception as e:
                logger.debug("Inline data parse failed: %s", e)

                # Error annotation
                _LLMObs.annotate(
                    span=tool_span,
                    output_data={"error": True, "extracted": False},
                    metadata={"error_message": str(e)}
                )

                return result
    else:
        # Fallback: existing code without instrumentation
        # [existing _extract_image_payload logic]
        pass
```

### Step 5: Refactor Endpoint to Use Workflow

**Modify**: `generate_gemini_image()` endpoint (line 494)

```python
@router_gemini.post("/gemini-generate-image", response_model=GeminiImageResponse)
async def generate_gemini_image(request: GeminiImageRequest):
    """Generate image with rich LLM Observability spans."""

    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    model_name = request.model or DEFAULT_MODEL
    width, height = _parse_size(request.size or "1024x1024")

    try:
        # Use workflow wrapper for hierarchical spans
        if _is_llmobs_enabled():
            with _LLMObs.workflow(name="image_generation") as workflow_span:
                # Workflow input
                _LLMObs.annotate(
                    span=workflow_span,
                    input_data={
                        "prompt": request.prompt,
                        "size": f"{width}x{height}",
                        "model": model_name
                    },
                    metadata={
                        "operation": "generate_image",
                        "model_family": "gemini" if "gemini" in model_name else "imagen"
                    }
                )

                # Step 1: Build payload (tool span)
                url, payload = _build_payload(model_name, request.prompt, width, height)

                # Step 2: API call (llm span)
                api_start = time.time()
                with _LLMObs.llm(
                    model_name=model_name,
                    name="gemini.api_call",
                    model_provider="google"
                ) as llm_span:
                    _LLMObs.annotate(
                        span=llm_span,
                        input_data=request.prompt,
                        metadata={
                            "endpoint": url,
                            "request_size_bytes": len(json.dumps(payload))
                        }
                    )

                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            url,
                            params={"key": api_key},
                            headers={"x-goog-api-key": api_key},
                            json=payload,
                        )

                        api_duration_ms = (time.time() - api_start) * 1000

                        if resp.status_code >= 400:
                            text = resp.text
                            logger.error("Gemini API error %s: %s", resp.status_code, text)
                            _LLMObs.annotate(
                                span=llm_span,
                                output_data={"error": True, "status_code": resp.status_code},
                                metadata={
                                    "error_message": text[:500],
                                    "api_latency_ms": api_duration_ms
                                }
                            )
                            raise HTTPException(
                                status_code=resp.status_code,
                                detail=f"Gemini API error ({resp.status_code}): {text[:400]}",
                            )

                        data = resp.json()

                        _LLMObs.annotate(
                            span=llm_span,
                            output_data={
                                "success": True,
                                "candidates_count": len(data.get("candidates", []))
                            },
                            metadata={
                                "http_status_code": resp.status_code,
                                "api_latency_ms": api_duration_ms
                            }
                        )

                # Step 3: Extract response (tool span)
                extracted = _extract_image_payload(data)
                image_b64 = extracted.get("data")
                mime_type = extracted.get("mime_type")

                # Workflow output
                _LLMObs.annotate(
                    span=workflow_span,
                    output_data={
                        "image_generated": bool(image_b64),
                        "image_bytes": len(image_b64) if image_b64 else 0,
                        "mime_type": mime_type,
                        "thought_present": bool(extracted.get("thought"))
                    },
                    metadata={
                        "total_duration_ms": (time.time() - workflow_span._start_time) * 1000,
                        "model": model_name,
                        "size": f"{width}x{height}"
                    }
                )

                # Build response_parts
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
        else:
            # Fallback to existing flat span implementation
            # [existing code from lines 509-584]
            pass

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Gemini image generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini image generation failed")
```

### Step 6: Apply Same Pattern to edit_gemini_image

Repeat workflow pattern for `edit_gemini_image()` endpoint (line 592) with:
- `LLMObs.workflow(name="image_editing")`
- Same child span structure
- Additional metadata for multi-turn context

## Expected Datadog View After Implementation

```
image_generation (workflow - 12.5s)
├── build_payload (tool - 0.2s)
│   Input: {prompt, dimensions, model}
│   Output: {endpoint, payload_size, aspect_ratio}
│   Metadata: {endpoint_type, modalities, safety_settings}
├── gemini.api_call (llm - 11.8s) ← This shows up as LLM call with cost/token metrics
│   Input: "Generate a Blue Morpho Butterfly"
│   Output: {success: true, candidates: 1}
│   Metadata: {api_latency_ms, http_status, response_size}
└── extract_image_payload (tool - 0.5s)
    Input: {candidates: 1, parts: 2}
    Output: {image_extracted: true, bytes: 125000, thought: true}
    Metadata: {extraction_time_ms, thought_signature, mime_type}
```

## Comparison: Before vs After

| Metric | Before (Sparse) | After (Rich) |
|--------|----------------|--------------|
| **Span Count** | 1 | 4+ |
| **Hierarchy** | Flat | 2 levels (workflow → tools/llm) |
| **Metadata Fields** | 3 | 15+ |
| **Input/Output** | Minimal | Comprehensive |
| **Timing Breakdown** | No | Yes (per step) |
| **Error Context** | Basic tags | Full error annotations |
| **Evaluations** | 0 | 0 (image gen doesn't support) |
| **Tokens** | 0 | 0 (image gen doesn't have tokens) |

## Implementation Checklist

- [ ] Add `LLMObs.workflow()` wrapper to both endpoints
- [ ] Instrument `_build_payload()` as `LLMObs.tool()`
- [ ] Wrap API call with `LLMObs.llm()` span
- [ ] Instrument `_extract_image_payload()` as `LLMObs.tool()`
- [ ] Add timing metrics (time.time() tracking)
- [ ] Add size metrics (bytes for payloads/images)
- [ ] Enrich metadata with config params
- [ ] Add comprehensive error annotations
- [ ] Test in Datadog - verify hierarchical view
- [ ] Apply pattern to `edit_gemini_image()` endpoint
- [ ] Document span structure for team reference

## Notes

- **Tokens/Cost**: Will remain N/A for image generation (no token-based pricing)
- **Evaluations**: Won't apply to image generation (text-only feature)
- **Agent Pattern**: Could wrap endpoints in `LLMObs.agent()` for top-level context
- **Performance**: Instrumentation adds <50ms overhead per span (negligible)
- **Compatibility**: Fallback to flat spans if LLMObs not available

## References

- Current Implementation: `src/gemini_service.py:494-698`
- OpenAI App Pattern: Screenshot showing hierarchical structure
- Datadog LLM Observability Docs: https://docs.datadoghq.com/llm_observability/
