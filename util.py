import base64
from pathlib import Path


def ensure_dir(d):
    Path.mkdir(d, exist_ok=True)

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")