"""Apply official RoboTwin compatibility changes inside the active environment."""
import importlib.util
from pathlib import Path
import sys
import imageio_ffmpeg

sapien_root = Path(importlib.util.find_spec("sapien").origin).parent
mplib_root = Path(importlib.util.find_spec("mplib").origin).parent
p = sapien_root / "wrapper/urdf_loader.py"
s = p.read_text()
s = s.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
s = s.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
p.write_text(s)
p = mplib_root / "planner.py"
s = p.read_text()
old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
assert old in s or new in s
p.write_text(s.replace(old, new))
# SAPIEN 3's legacy SapienRenderer ignores device arguments; bind the actual RenderSystem.
p = sapien_root / "wrapper/scene.py"
s = p.read_text()
if "import os\n" not in s:
    s = s.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n")
s = s.replace("sapien.render.RenderSystem()]", 'sapien.render.RenderSystem(device=os.environ.get("SAPIEN_RENDER_DEVICE"))]')
p.write_text(s)
p = Path(sys.executable).parent / "ffmpeg"
if not p.exists():
    p.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
print("RoboTwin compatibility adjustments applied.")
