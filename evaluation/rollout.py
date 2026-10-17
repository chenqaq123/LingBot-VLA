"""Call the pinned upstream RoboTwin evaluator, preserving its scene and control loop."""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--setting', default='demo_randomized', choices=['demo_clean', 'demo_randomized'])
    p.add_argument('--episodes', default=1, type=int)
    p.add_argument('--instruction-type', choices=['seen', 'unseen'])
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    faulthandler.dump_traceback_later(120, repeat=True)
    if args.episodes < 1:
        p.error('episodes must be positive')
    root = Path(os.environ['ROBOTWIN_ROOT'])
    upstream = Path(args.upstream)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root/'description/utils'), str(upstream)]
    from deploy.websocket_client_policy import WebsocketClientPolicy
    source = upstream/'experiment/robotwin/eval_policy_client_lingbotvla.py'
    spec = importlib.util.spec_from_file_location('official_robotwin_eval', source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((root/f'task_config/{args.setting}.yml').read_text())
    embodiment = root/'assets/embodiments/aloha-agilex'
    robot_cfg = yaml.safe_load((embodiment/'config.yml').read_text())
    camera_cfg = yaml.safe_load((root/'task_config/_camera_config.yml').read_text())[cfg['camera']['head_camera_type']]
    cfg.update(task_name=args.task, task_config=args.setting, policy_name='lingbot_official',
               ckpt_setting='official_inference', left_robot_file=str(embodiment), right_robot_file=str(embodiment),
               left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg,
               dual_arm_embodied=True, head_camera_h=camera_cfg['h'], head_camera_w=camera_cfg['w'],
               eval_video_log=True, eval_video_save_dir=out)
    task = official.class_decorator(args.task)
    # Trace setup/expert phases without changing inputs, RNG, or success checks.
    original_setup = task.setup_demo
    original_play = task.play_once
    def traced_setup(*positional, **keywords):
        started = time.monotonic()
        print('SCENE_SETUP_BEGIN', keywords.get('seed'), flush=True)
        value = original_setup(*positional, **keywords)
        print('SCENE_SETUP_READY', keywords.get('seed'), round(time.monotonic()-started, 2), flush=True)
        return value
    def traced_play(*positional, **keywords):
        started = time.monotonic()
        print('EXPERT_CHECK_BEGIN', flush=True)
        value = original_play(*positional, **keywords)
        print('EXPERT_CHECK_END', bool(task.plan_success), round(time.monotonic()-started, 2), flush=True)
        return value
    task.setup_demo = traced_setup
    task.play_once = traced_play
    client = WebsocketClientPolicy(host='127.0.0.1', port=args.port)
    records = []
    class RecordedClient:
        def infer(self, observation):
            if observation.get('reset'):
                records.append({'episode': len(records), 'policy_calls': 0, 'action_shapes': []})
            elif records[-1]['policy_calls'] == 0:
                record = records[-1]
                record['instruction'] = observation['task']
                record['initial_state'] = np.asarray(observation['observation.state']).tolist()
                record['image_shapes'] = {k: list(v.shape) for k, v in observation.items() if k.startswith('observation.images.')}
                record['head_rgb_sha256'] = hashlib.sha256(observation['observation.images.cam_high'].tobytes()).hexdigest()
                # Keep one real observation for deterministic preprocessing/inference checks.
                np.savez(out/f'observation_{len(records)-1:03d}.npz', **observation)
            result = client.infer(observation)
            if not observation.get('reset'):
                action = np.asarray(result['action'])
                if action.shape != (50, 14) or not np.isfinite(action).all():
                    raise ValueError(f'Invalid official action block: {action.shape}')
                records[-1]['policy_calls'] += 1
                records[-1]['action_shapes'].append(list(action.shape))
                if records[-1]['policy_calls'] == 1:
                    np.save(out/f'first_actions_{len(records)-1:03d}.npy', action)
                (out/'requests.json').write_text(json.dumps(records, indent=2))
            return result
    # The pinned LingBot launcher selects policy/ACT/deploy_policy.yml for both settings.
    deploy_config = root/'policy/ACT/deploy_policy.yml'
    instruction_type = args.instruction_type or yaml.safe_load(deploy_config.read_text())['instruction_type']
    report = {'implementation': 'official', 'task': args.task, 'setting': args.setting,
              'requested_episodes': args.episodes, 'instruction_type': instruction_type,
              'instruction_config': str(deploy_config),
              'renderer': 'upstream default ray tracing; no raster override',
              'evaluator': str(source), 'evaluator_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'start_seed': 100000*(1+args.seed), 'status': 'running'}
    (out/'result.json').write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    try:
        final_seed, successes = official.eval_policy(args.task, task, cfg, RecordedClient(),
            report['start_seed'], test_num=args.episodes, video_size=f"{camera_cfg['w']}x{camera_cfg['h']}",
            instruction_type=instruction_type, usr_args={'robo_name': 'robotwin'})
        report.update(status='complete', episodes=task.test_num, successes=successes, final_seed=final_seed)
    except BaseException as exc:
        report.update(status='error', error=repr(exc), episodes=task.test_num)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        report.update(elapsed_seconds=time.monotonic()-started, requests=records)
        (out/'result.json').write_text(json.dumps(report, indent=2))
    print('OFFICIAL_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
