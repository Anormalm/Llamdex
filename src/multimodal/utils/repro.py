import json
import os
import platform
import random
from datetime import datetime, timezone

import numpy as np
import torch
import transformers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_config(config: dict, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def save_versions(run_dir: str) -> None:
    versions = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    with open(os.path.join(run_dir, "versions.json"), "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, sort_keys=True)

