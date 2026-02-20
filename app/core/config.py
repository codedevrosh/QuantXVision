from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50_processed.csv"
LSTM_MODEL_DIR = PROJECT_ROOT / "models" / "lstm"
PROPHET_MODEL_DIR = PROJECT_ROOT / "models" / "prophet"