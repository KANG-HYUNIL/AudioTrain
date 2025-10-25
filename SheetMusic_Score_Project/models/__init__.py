"""
Model registry for student and teacher builders.

Usage:
	from models import build_student_model, build_teacher_model

To add a new model, create a file (e.g., student_mymodel.py or teacher_mymodel.py)
that imports `register_student` or `register_teacher` and registers a callable
`builder(**kwargs) -> nn.Module`.
"""

from __future__ import annotations
from typing import Callable, Dict

import importlib
# from . import student_mobilenet  # noqa: F401
# from . import teacher_passt  # noqa: F401
# from . import teacher_vit  # noqa: F401

# Registries
_STUDENT_BUILDERS: Dict[str, Callable[..., object]] = {}
_TEACHER_BUILDERS: Dict[str, Callable[..., object]] = {}


def register_student(name: str):
	def deco(fn: Callable[..., object]):
		_STUDENT_BUILDERS[name.lower()] = fn
		return fn
	return deco


def register_teacher(name: str):
	def deco(fn: Callable[..., object]):
		_TEACHER_BUILDERS[name.lower()] = fn
		return fn
	return deco


# Eagerly import known model modules to populate registry
for _mod in [
	'models.student_mobilenet',
	'models.teacher_passt',
	'models.teacher_vit',
]:
	try:
		importlib.import_module(_mod)
	except Exception:
		pass


def build_student_model(arch: str, **kwargs):
	key = arch.lower()
	if key not in _STUDENT_BUILDERS:
		raise ValueError(f"Unknown student arch: {arch}. Available: {sorted(_STUDENT_BUILDERS.keys())}")
	return _STUDENT_BUILDERS[key](**kwargs)


def build_teacher_model(name: str, **kwargs):
	key = name.lower()
	if key not in _TEACHER_BUILDERS:
		raise ValueError(f"Unknown teacher name: {name}. Available: {sorted(_TEACHER_BUILDERS.keys())}")
	return _TEACHER_BUILDERS[key](**kwargs)
