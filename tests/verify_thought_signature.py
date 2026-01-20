
from src.gemini_service import GeminiImagePart, _build_multi_turn_payload, GeminiImageMessage

try:
    print("Testing GeminiImagePart with thought field...")
    part = GeminiImagePart(image_base64="data", thought=True)
    if part.thought is True:
        print("SUCCESS: 'thought' field accepted")
    else:
        print("FAILURE: 'thought' field not set")

    print("\nTesting payload construction...")
    msgs = [
        GeminiImageMessage(role="model", parts=[part])
    ]
    url, payload, _ = _build_multi_turn_payload("gemini-1.5-flash", msgs)
    
    part_payload = payload["contents"][0]["parts"][0]
    if part_payload.get("thought") is True:
        print("SUCCESS: 'thought' field included in payload")
    else:
        print(f"FAILURE: 'thought' field missing from payload: {part_payload}")

except Exception as e:
    print(f"ERROR: {e}")
