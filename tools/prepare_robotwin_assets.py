"""Write small robot path configs; meshes/textures stay in shared storage."""
import os
from pathlib import Path
import yaml


def prepare():
    root = Path(__file__).resolve().parents[1] / 'RoboTwin'
    assets = Path(os.environ.get('ROBOTWIN_ASSETS', '/srv/common_dataset/RoboTwin2.0/assets')).resolve()
    for name in ('objects', 'background_texture', 'embodiments'):
        if not (assets / name).is_dir():
            raise FileNotFoundError(assets / name)

    def write_if_changed(path, text):
        if not path.exists() or path.read_text() != text:
            path.write_text(text)

    for robot in (assets / 'embodiments').iterdir():
        if not robot.is_dir() or not (robot / 'config.yml').is_file():
            continue
        destination = root / 'assets/embodiments' / robot.name
        destination.mkdir(parents=True, exist_ok=True)
        cfg = yaml.safe_load((robot / 'config.yml').read_text())
        for key in ('urdf_path', 'srdf_path'):
            if cfg.get(key):
                cfg[key] = str(robot / cfg[key])
        write_if_changed(destination / 'config.yml', yaml.safe_dump(cfg, sort_keys=False))
        for template in robot.glob('*_tmp.yml'):
            content = template.read_text().replace('${ASSETS_PATH}/assets', str(assets)).replace('$ASSETS_PATH/assets', str(assets))
            write_if_changed(destination / template.name.replace('_tmp.yml', '.yml'), content)


if __name__ == '__main__':
    prepare()
