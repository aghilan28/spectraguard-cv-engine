import numpy as np
from typing import Any

def convert_numpy_types(obj: Any) -> Any:
    """Recursively cast NumPy data types to standard Python primitives."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj
