import os
import sys
from pathlib import Path

# Explicitly guarantee the project root directory is registered globally during test discovery
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
