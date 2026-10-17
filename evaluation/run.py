"""Launch official LingBot inference plus its original RoboTwin evaluation loop.

Provide full HF weights and the corresponding training YAML for fine-tuned checkpoints.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'lingbot-vla-v2'
BASE_MODEL = Path('/srv/common_model/huggingface/lingbot-vla-v2-6b')


def prepare_checkpoint(out, source, config_path, tokenizer, upstream):
    if source.is_file() or (source/'adapter.safetensors').exists():
        raise ValueError('Output-head adapters are unsupported. Supply a full checkpoint.')
    if config_path is None:
        saved = source.parent.parent.parent/'lingbotvla_cli.yaml'
        if saved.is_file():
            config_path = saved
        elif source.resolve() == BASE_MODEL.resolve():
            config_path = upstream/'configs/vla/robotwin/robotwin.yaml'
        else:
            raise ValueError('Full fine-tuned checkpoints require --training-config or their saved lingbotvla_cli.yaml')
    config = copy.deepcopy(yaml.safe_load(config_path.read_text()))
    is_template = config_path == upstream/'configs/vla/robotwin/robotwin.yaml'
    config['model']['tokenizer_path'] = str(tokenizer)
    # Raw YAML lists contain mappings; saved official CLI YAML contains strings.
    for key in ['joints', 'norm_type']:
        config['data'][key] = [repr(x) if isinstance(x, dict) else x for x in config['data'][key]]
    norm_value = config['data'].get('norm_stats_file')
    if norm_value:
        norm = Path(norm_value)
        if not norm.is_absolute():
            candidates = [config_path.parent/norm, upstream/norm]
            norm = next((x for x in candidates if x.is_file()), candidates[0])
    else:
        norm = upstream/'assets/norm_stats/robotwin.json'
    if not norm.is_file():
        raise FileNotFoundError(f'Checkpoint normalization file is missing: {norm}')
    snapshot = out/'model'
    snapshot.mkdir(parents=True)
    # CIFS does not support symlinks; keep the weight view on local disk.
    runtime_snapshot = Path(tempfile.mkdtemp(prefix='lingbo-official-model-'))
    weights = runtime_snapshot/'checkpoints/global_step_000000/hf_ckpt'
    weights.mkdir(parents=True)
    norm_copy = snapshot/'norm_stats.json'
    norm_copy.write_bytes(norm.read_bytes())
    config['data']['norm_stats_file'] = str(norm_copy)
    serialized = yaml.safe_dump(config, sort_keys=False)
    (snapshot/'lingbotvla_cli.yaml').write_text(serialized)
    (runtime_snapshot/'lingbotvla_cli.yaml').write_text(serialized)
    shards = sorted(source.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'No full weights found under {source}')
    for shard in shards:
        (weights/shard.name).symlink_to(shard.resolve())
    return weights, {'source_weights': str(source), 'source_config': str(config_path),
                     'config_origin': 'official RoboTwin template for base pretrained weights' if is_template else 'checkpoint training configuration',
                     'base_model_not_robotwin_posttrained': source.resolve() == BASE_MODEL.resolve(),
                     'normalization_source': str(norm), 'normalization_sha256': hashlib.sha256(norm_copy.read_bytes()).hexdigest(),
                     'effective_image_size': config['data'].get('img_size', 256)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--tasks', nargs='+', default=['click_bell'])
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--instruction-type', choices=['seen', 'unseen'])
    p.add_argument('--setting', default='demo_randomized', choices=['demo_clean', 'demo_randomized'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model-path', type=Path, default=BASE_MODEL)
    p.add_argument('--training-config', type=Path)
    p.add_argument('--tokenizer-path', type=Path, default=Path('/srv/common_model/huggingface/Qwen3-VL-4B-Instruct'))
    p.add_argument('--model-gpus', nargs='+', default=['3', '5'])
    p.add_argument('--sim-gpu', default='6')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--timeout', type=int, default=7200)
    p.add_argument('--continue-on-error', action='store_true', help='Record infrastructure errors and continue other tasks')
    args = p.parse_args()
    if args.episodes < 1 or len(args.model_gpus) not in (1, 2):
        p.error('Use positive episodes and one or two model GPUs')
    if len(set(args.model_gpus+[args.sim_gpu])) != len(args.model_gpus)+1:
        p.error('Model GPUs and simulator GPU must be distinct')
    if args.precision == 'fp32' and len(args.model_gpus) == 1:
        required = [31000]
    else:
        required = [22500, 4500] if args.precision == 'fp32' else [15500, 2500]
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    available = {k.strip(): int(v) for k, v in (line.split(',') for line in free.strip().splitlines())}
    for gpu, minimum in list(zip(args.model_gpus, required))+[(args.sim_gpu, 6500)]:
        if available.get(gpu, 0) < minimum:
            raise RuntimeError(f'GPU {gpu}: need {minimum} MiB free, have {available.get(gpu, 0)}')
    manifest = json.loads((ROOT/'configs/robotwin_tasks.json').read_text())
    if set(args.tasks)-set(manifest['tasks']) or len(set(args.tasks)) != len(args.tasks):
        p.error('Unknown or duplicate tasks')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    upstream = UPSTREAM
    weights, provenance = prepare_checkpoint(out, args.model_path.resolve(), args.training_config,
                                             args.tokenizer_path.resolve(), upstream)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    env = {**os.environ, 'PYTHONPATH': f'{ROOT}:{upstream}', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    for key in ['NO_PROXY', 'no_proxy']:
        env[key] = env.get(key, '')+',127.0.0.1,localhost'
    for key in ['LINGBO_STRICT_FP32', 'NVIDIA_TF32_OVERRIDE', 'LINGBO_UPSTREAM']:
        env.pop(key, None)
    source_commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    report = {'upstream_commit': source_commit, 'upstream_source': str(upstream), 'model': provenance,
              'tasks': args.tasks, 'setting': args.setting, 'episodes_per_task': args.episodes,
              'precision': args.precision, 'model_gpus': args.model_gpus, 'sim_gpu': args.sim_gpu,
              'status': 'starting', 'results': [], 'errors': [],
              'scope': 'Official inference methods and evaluator; checkpoint loading/device placement adapted for 24 GB GPUs. Not a full benchmark unless run with required tasks/episodes.'}
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2))
    save()
    ready = out/'model_ready.json'
    command = [sys.executable, '-u', '-m', 'evaluation.policy', '--checkpoint', str(weights),
               '--port', str(port), '--ready-file', str(ready), '--precision', args.precision]
    if len(args.model_gpus) == 2:
        command += ['--frontend-device', 'cuda:1']
    processes = []
    try:
        with (out/'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=upstream, env={**env, 'CUDA_VISIBLE_DEVICES': ','.join(args.model_gpus)},
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            deadline = time.monotonic()+900
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f'Official policy startup failed; see {out/"server.log"}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Official policy startup timed out')
                time.sleep(1)
            time.sleep(1)
            report['status'] = 'evaluating'; save()
            print('OFFICIAL_SERVER_READY', str(ready), flush=True)
            for task in args.tasks:
                task_out = out/task
                with (out/f'{task}.log').open('w') as simlog:
                    cmd = ['bash', str(ROOT/'run_robotwin.sh'), 'python', '-u', str(ROOT/'evaluation/rollout.py'),
                           '--upstream', str(upstream), '--task', task, '--setting', args.setting,
                           '--episodes', str(args.episodes), '--seed', str(args.seed), '--port', str(port), '--output', str(task_out)]
                    if args.instruction_type:
                        cmd += ['--instruction-type', args.instruction_type]
                    proc = subprocess.Popen(cmd, env={**env, 'ROBOTWIN_GPU': args.sim_gpu,
                                            'PYTHONPATH': '/tmp/lingbo-official-sim-deps:'+env['PYTHONPATH']}, stdout=simlog,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append(proc)
                    try:
                        result = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        result = 'timeout'
                    if result != 0:
                        report['errors'].append({'task': task, 'exit_code': result, 'log': str(out/f'{task}.log')})
                        save()
                        if args.continue_on_error:
                            print('TASK_ERROR', task, result, flush=True)
                            continue
                        raise RuntimeError(f'Official simulator failed for {task}; see {out/f"{task}.log"}')
                report['results'].append(json.loads((task_out/'result.json').read_text()))
                save(); print('TASK_COMPLETE', task, flush=True)
            report['status'] = 'incomplete' if report['errors'] else 'complete'; save()
    except BaseException as exc:
        report.update(status='error', error=repr(exc)); save()
        raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()

    if report['status'] == 'incomplete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
