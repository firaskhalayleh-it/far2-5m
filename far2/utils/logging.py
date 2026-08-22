import json
import os
from typing import Any


class JsonlLogger:
    """JSONL writer; one record per event, flushed immediately. Truncates any
    existing file: a rerun must never interleave old and new evidence."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._fh = open(path, "w")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_jsonable) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _jsonable(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.item() if x.numel() == 1 else x.tolist()
    except ImportError:
        pass
    return str(x)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
