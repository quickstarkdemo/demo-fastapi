
from src.gemini_service import GeminiImageMessage, GeminiImagePart, _build_multi_turn_payload
from pydantic import ValidationError

try:
    print("Testing role validation...")
    # Test valid roles
    GeminiImageMessage(role="user", parts=[GeminiImagePart(text="hi")])
    GeminiImageMessage(role="assistant", parts=[GeminiImagePart(text="hello")])
    GeminiImageMessage(role="model", parts=[GeminiImagePart(text="hello")])
    print("SUCCESS: Valid roles accepted")

    # Test invalid role
    try:
        GeminiImageMessage(role="invalid", parts=[GeminiImagePart(text="hi")])
        print("FAILURE: Invalid role accepted")
    except ValidationError:
        print("SUCCESS: Invalid role rejected")

    print("\nTesting payload construction...")
    msgs = [
        GeminiImageMessage(role="user", parts=[GeminiImagePart(text="hi")]),
        GeminiImageMessage(role="assistant", parts=[GeminiImagePart(text="hello")]),
    ]
    url, payload, _ = _build_multi_turn_payload("gemini-1.5-flash", msgs)
    
    contents = payload["contents"]
    if contents[0]["role"] == "user" and contents[1]["role"] == "model":
        print("SUCCESS: 'assistant' mapped to 'model'")
    else:
        print(f"FAILURE: Incorrect mapping: {contents}")

except Exception as e:
    print(f"ERROR: {e}")
