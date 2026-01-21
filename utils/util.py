import base64
from pathlib import Path
import re


def ensure_dir(d):
    Path.mkdir(d, exist_ok=True)

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def natural_key(s):
    """Sort key that handles numbers inside strings."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]