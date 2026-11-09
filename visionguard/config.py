"""Project-wide constants shared by training and serving."""
from pathlib import Path

PROJECT_NAME = "23L-0517_A2_Ops"
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CACHE_DIR = ROOT / "data" / "cache"

# Folder name on disk -> label. Positive class (1) is "AI-generated".
CLASS_FOLDERS = {"RealArt": 0, "AiArtData": 1}
CLASS_NAMES = {0: "Real", 1: "AI-generated"}

SEED = 42
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH = 32
