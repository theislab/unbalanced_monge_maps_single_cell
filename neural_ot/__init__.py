from importlib.metadata import version

from . import data, models, utils

__all__ = ["data", "models", "utils"]

__version__ = version("Neural-OT")
