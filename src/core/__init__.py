"""Core modules for BarnSight Edge.

Exports StreamHandler for camera stream management.
Other modules (logger, queue, region_tracker) are
imported directly by their submodules.
"""

__all__ = ["StreamHandler"]


def __getattr__(name: str):
  """Lazy import to avoid pulling in cv2 at module load time."""
  if name == "StreamHandler":
    from .stream_handler import StreamHandler
    return StreamHandler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
