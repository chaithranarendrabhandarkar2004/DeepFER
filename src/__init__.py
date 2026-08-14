import sys
from pathlib import Path

# Ensure project root directory is in sys.path for smooth imports across scripts
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
