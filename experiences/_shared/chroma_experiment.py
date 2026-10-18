"""Import the frozen experiment 08 helpers, outside the runtime package."""
from importlib import import_module

_module = import_module("experiences.08.run")
for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)
