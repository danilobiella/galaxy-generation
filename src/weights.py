"""Params-only inference artifact, stored as flat safetensors."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from safetensors.numpy import load_file, save_file

SEPARATOR = "/"


def flatten_params(params: Mapping[str, Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    flat: Dict[str, np.ndarray] = {}
    for module_name, module_params in params.items():
        for param_name, value in module_params.items():
            if SEPARATOR in param_name:
                raise ValueError(
                    f"Parameter name {param_name!r} in module {module_name!r} "
                    f"contains {SEPARATOR!r}, which would break unflattening."
                )
            flat[f"{module_name}{SEPARATOR}{param_name}"] = np.asarray(value)
    return flat


def unflatten_params(flat: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    params: Dict[str, Dict[str, np.ndarray]] = {}
    for key, value in flat.items():
        module_name, param_name = key.rsplit(SEPARATOR, 1)
        params.setdefault(module_name, {})[param_name] = value
    return params


def save_params(
    params: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    dtype: Optional[str] = None,
) -> Path:
    """Write params to `path`. `dtype` (e.g. "float16") casts on the way out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = flatten_params(params)
    if dtype is not None:
        flat = {k: v.astype(dtype) for k, v in flat.items()}

    save_file(flat, str(path))
    return path


def load_params(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load params as the nested dict haiku's `apply` expects."""
    return unflatten_params(load_file(str(path)))
