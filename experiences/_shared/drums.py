"""Frozen rhythm prototype used by experiment 02."""
from importlib import import_module
_module = import_module("experiences.02.drum_similarity")
for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)
