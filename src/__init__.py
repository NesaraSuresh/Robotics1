"""G82 Track A pipeline package.

Selects a MuJoCo OpenGL backend *before* mujoco is imported anywhere:
  * an explicit MUJOCO_GL environment variable always wins;
  * headless Linux (no DISPLAY / WAYLAND_DISPLAY) -> OSMesa software rendering;
  * desktops (Linux with a display, macOS, Windows) -> MuJoCo's default (GLFW).
Always `import src...` before `import mujoco` in scripts.
"""
import os as _os
import sys as _sys

if "MUJOCO_GL" not in _os.environ and _sys.platform.startswith("linux") \
        and not (_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY")):
    _os.environ["MUJOCO_GL"] = "osmesa"
if _os.environ.get("MUJOCO_GL") == "osmesa":
    _os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
