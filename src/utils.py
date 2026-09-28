from __future__ import annotations

import os
import sys
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_api_key() -> str:
    """Read XENO_CANTO_API_KEY from the environment or a local .env file."""
    key = os.environ.get("XENO_CANTO_API_KEY")
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("XENO_CANTO_API_KEY") and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit(
            "ERROR: no Xeno-canto API key found.\n"
            "  Get one (free) at https://xeno-canto.org/account, then either:\n"
            '    export XENO_CANTO_API_KEY="your-key"\n'
            "  or copy .env.example to .env and paste your key there.")
    return key


def set_seed(seed: int = 1337) -> None:
    """Seed python / numpy / torch (torch seeded only if installed)."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device():
    """Prefer CUDA, then Apple MPS (Adi's machine), then CPU. Imports torch lazily."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
