"""Run one scripted RoboTwin task and save video; no model or training data needed."""
import argparse
import faulthandler
import importlib
import json
import os
from pathlib import Path
import sys
import time

root = Path(os.environ['ROBOTWIN_ROOT'])
os.chdir(root)
sys.path.insert(0, str(root))

def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='click_bell')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render-mode', choices=['raster', 'raytrace'], default='raster')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'outputs/simulator_smoke'))
    parser.add_argument('--viewer', action='store_true', help='Requires a working graphical display.')
    options = parser.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    import sapien.core as sapien

    print('GPU:', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', ''), flush=True)
    config = yaml.safe_load((root / 'task_config/demo_clean.yml').read_text())
    embodiment = root / 'assets/embodiments/aloha-agilex'
    robot_config = yaml.safe_load((embodiment / 'config.yml').read_text())
    output = Path(options.output) / (options.task + '_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    config.update(task_name=options.task, task_config='installation_smoke_only',
                  save_path=str(output), save_data=False, collect_data=False,
                  render_freq=20 if options.viewer else 0, save_freq=15,
                  left_robot_file=str(embodiment), right_robot_file=str(embodiment),
                  left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                  dual_arm_embodied=True, embodiment_name='aloha-agilex', need_plan=True)
    task_class = getattr(importlib.import_module('envs.' + options.task), options.task)
    task = task_class()
    if options.render_mode == 'raster':
        original_setup_scene = task.setup_scene
        def setup_preview_scene(**kwargs):
            original_setup_scene(**kwargs)
            sapien.render.set_camera_shader_dir('default')
        task.setup_scene = setup_preview_scene
    writer = imageio.get_writer(output / 'demo.mp4', fps=250/15, codec='libx264', quality=7)
    frames = 0
    started = time.monotonic()
    first_frame_std = None

    def record():
        nonlocal frames, first_frame_std
        frame = task.get_obs()['observation']['head_camera']['rgb']
        if frames == 0:
            imageio.imwrite(output / 'preview.png', frame)
            first_frame_std = float(np.std(frame))
        writer.append_data(frame)
        frames += 1
        if frames % 25 == 0:
            print('VIDEO_FRAMES', frames, 'elapsed', round(time.monotonic()-started, 1), flush=True)

    try:
        task.setup_demo(now_ep_num=0, seed=options.seed, **config)
        task._take_picture = record
        record()
        task.play_once()
        record()
        result = {'task': options.task, 'seed': options.seed, 'controller': 'official scripted expert',
                  'training_data': False, 'render_mode': options.render_mode, 'plan_success': bool(task.plan_success),
                  'task_success': bool(task.check_success()), 'frames': frames,
                  'first_frame_std': first_frame_std, 'video': str(output / 'demo.mp4'),
                  'gpu': torch.cuda.get_device_name(0)}
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        if not result['plan_success'] or not result['task_success'] or frames < 2 or first_frame_std < 1:
            raise RuntimeError('Smoke task did not complete successfully; inspect result.json and video.')
    finally:
        writer.close()
        faulthandler.cancel_dump_traceback_later()
        if hasattr(task, 'scene'):
            task.close_env()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
