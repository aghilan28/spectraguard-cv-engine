import sys
from pathlib import Path
import pytest

root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"[TEST RUNNER] Path map established. Project Anchor: {root_dir}")
print("[TEST RUNNER] Executing targeted Phase 6 Unit Tests...")

# Run pytest programmatically on the specific test module path
exit_code = pytest.main(["-v", "tests/training/test_dataset_pipeline.py"])

# FIX: Pass the exit code as a standard positional argument
sys.exit(exit_code)
