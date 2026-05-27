from .config import FocusDiffConfig


def __getattr__(name):
    if name == "FocusDiff":
        from .pipeline import FocusDiff

        return FocusDiff
    raise AttributeError(name)

__all__ = ["FocusDiff", "FocusDiffConfig"]
