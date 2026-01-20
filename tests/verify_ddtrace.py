
try:
    from ddtrace import tracer
    with tracer.trace("test", resource="test") as span:
        if hasattr(span, "set_tag"):
            print("SUCCESS: span.set_tag exists")
            span.set_tag("test_key", "test_value")
        else:
            print("FAILURE: span.set_tag does not exist")
        
        if hasattr(span, "set_tag_str"):
            print("NOTE: span.set_tag_str exists (but we are removing it)")
        else:
            print("NOTE: span.set_tag_str does not exist")

except ImportError:
    print("WARNING: ddtrace not installed")
except Exception as e:
    print(f"ERROR: {e}")
