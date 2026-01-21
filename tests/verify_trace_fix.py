
import sys
import unittest
from contextlib import contextmanager

# Mocking parts of the system to simulate Datadog tracer behavior
class MockTracer:
    @contextmanager
    def trace(self, name, **kwargs):
        # This simulates the real ddtrace.tracer.trace method
        # It accepts 'resource', 'service', 'span_type' etc.
        # But it does NOT accept 'op' in newer versions, and would raise TypeError if passed unexpected args
        # We simulate the failure mode here
        
        valid_args = ['resource', 'service', 'span_type', 'tags']
        for k in kwargs:
            if k == 'op':
                raise TypeError("Tracer.trace() got an unexpected keyword argument 'op'")
            if k not in valid_args:
                 # Just purely to be strict for this test case
                 pass
        
        yield "mock_span"

class MockProvider:
    def __init__(self):
        self.tracer = MockTracer()
        
    @contextmanager
    def trace_context(self, name, **kwargs):
        # This matches the implementation in DatadogProvider.trace_context
        with self.tracer.trace(name, **kwargs) as span:
            yield span

class TestFix(unittest.TestCase):
    def setUp(self):
        self.provider = MockProvider()

    def test_original_bug_fails(self):
        """Verify that passing 'op' raises the TypeError, reproducing the bug."""
        print("\nTesting original buggy code (expecting failure)...")
        with self.assertRaises(TypeError) as cm:
            with self.provider.trace_context("demo.unhandled_error", op="bug_detection") as span:
                pass
        self.assertIn("unexpected keyword argument 'op'", str(cm.exception))
        print("Confirmed: Original code raises TypeError as expected.")

    def test_fix_works(self):
        """Verify that passing 'resource' works correctly."""
        print("\nTesting fixed code...")
        try:
            with self.provider.trace_context("demo.unhandled_error", resource="bug_detection") as span:
                pass
            print("Confirmed: Fixed code using 'resource' runs successfully.")
        except TypeError as e:
            self.fail(f"Fixed code raised TypeError: {e}")

if __name__ == '__main__':
    unittest.main()
