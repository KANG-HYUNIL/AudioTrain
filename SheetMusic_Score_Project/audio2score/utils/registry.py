"""Lightweight name→factory registry utility for backend plug-ins.

Use this to register detection/separation/transcription backends by name and
instantiate them later via a common interface.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional


class Registry:
    """Simple name -> factory registry.

    Example:
        REG = Registry("detector")
        @REG.register("passt")
        def build_passt(**kwargs): ...
        REG.get("passt")(...)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._store: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None):
        def _decorator(fn: Callable[..., Any]):
            key = name or fn.__name__
            if key in self._store:
                raise KeyError(f"{self._name} registry already has '{key}'")
            self._store[key] = fn
            return fn
        return _decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._store:
            known = ", ".join(sorted(self._store.keys()))
            raise KeyError(f"{self._name} backend '{name}' not found. Known: [{known}]")
        return self._store[name]

    def list(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._store)
