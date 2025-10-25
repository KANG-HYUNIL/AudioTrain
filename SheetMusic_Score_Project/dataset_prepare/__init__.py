from typing import Callable, Dict

# from . import nsynth

_DATASET_PREPARERS: Dict[str, Callable] = {}

def register_preparer(name: str):
    def deco(fn: Callable):
        _DATASET_PREPARERS[name.lower()] = fn
        return fn
    return deco

def prepare_dataset(name: str, **kwargs):
    key = name.lower()
    if key not in _DATASET_PREPARERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_PREPARERS.keys())}")
    return _DATASET_PREPARERS[key](**kwargs)

# Eager import all dataset modules to populate registry
for mod in [
    "dataset_prepare.nsynth",
    "dataset_prepare.tinysol",
    "dataset_prepare.irmas",
    "dataset_prepare.openmic",
]:
    try:
        __import__(mod)
    except Exception:
        pass