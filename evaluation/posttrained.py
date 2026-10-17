"""Run all 50 tasks (randomized by default) with pretrained, posttrained, or both models."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from evaluation.render_results import render

ROOT = Path(__file__).resolve().parents[1]


def export_results(work, output, tasks):
    report = json.loads((work / 'report.json').read_text())
    if report['status'] != 'complete' or report['errors']:
        raise ValueError('Evaluation is incomplete; diagnostic files are retained')
    model_label = ('LingBot-VLA-v2-6B 预训练版'
                   if report['model']['base_model_not_robotwin_posttrained']
                   else 'LingBot-VLA-v2-6B RoboTwin 后训练版')
    rows, sources = [], []
    for task in tasks:
        result = json.loads((work / task / 'result.json').read_text())
        videos = list((work / task).glob('*.mp4'))
        if (result['status'] != 'complete' or result['episodes'] != 1
                or result['successes'] not in (0, 1) or len(result['requests']) != 1
                or len(videos) != 1 or videos[0].stat().st_size == 0):
            raise ValueError(f'Invalid result or missing video: {task}')
        request = result['requests'][0]
        rows.append(dict(task=task, model=model_label, setting=result['setting'], success=result['successes'],
                         policy_calls=request['policy_calls'],
                         elapsed_seconds=round(result['elapsed_seconds'], 2),
                         episode_seed=result['final_seed'] - 1,
                         instruction=request['instruction'], video=f'videos/{task}.mp4'))
        sources.append(videos[0])
    if not rows:
        raise ValueError('No task results')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'videos').mkdir()
    for row, source in zip(rows, sources):
        shutil.copyfile(source, output / row['video'])
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pretrained', 'posttrained', 'both'], default='posttrained')
    parser.add_argument('--setting', choices=['demo_clean', 'demo_randomized'], default='demo_randomized')
    parser.add_argument('--model-gpus', nargs='+', default=['3', '2'])
    parser.add_argument('--sim-gpu', default='0')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds allowed per task')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    scene = args.setting.removeprefix('demo_')
    output = (args.output or ROOT / 'outputs' /
              f'{args.model}_{scene}50_video_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if output.exists():
        parser.error(f'Output already exists: {output}')
    tasks = json.loads((ROOT / 'configs/robotwin_tasks.json').read_text())['tasks']
    checkpoint = ROOT / 'models/lingbot-vla-v2-6b-robotwin'
    models = ['pretrained', 'posttrained'] if args.model == 'both' else [args.model]
    failures = []
    for model in models:
        destination = output / model if args.model == 'both' else output
        work = Path(tempfile.mkdtemp(prefix=f'lingbo-{model}-{scene}50-'))
        run = work / 'run'
        weights = (Path('/srv/common_model/huggingface/lingbot-vla-v2-6b')
                   if model == 'pretrained' else checkpoint / 'checkpoints/global_step_50000/hf_ckpt')
        print(f'Model: {model}\nResults: {destination}\nTemporary diagnostics: {run}', flush=True)
        command = [sys.executable, '-u', '-m', 'evaluation.run',
                   '--model-path', str(weights),
                   '--tasks', *tasks, '--episodes', '1', '--instruction-type', 'unseen',
                   '--setting', args.setting, '--seed', '0', '--precision', 'fp32',
                   '--model-gpus', *args.model_gpus, '--sim-gpu', args.sim_gpu,
                   '--continue-on-error', '--timeout', str(args.timeout), '--output', str(run)]
        if model == 'posttrained':
            command += ['--training-config', str(checkpoint / 'lingbotvla_cli.yaml')]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            export_results(run, destination, tasks)
        except KeyboardInterrupt:
            print(f'Interrupted. Diagnostics retained at: {run}', file=sys.stderr, flush=True)
            raise
        except Exception as exc:
            failures.append(model)
            print(f'{model} failed: {exc}. Diagnostics retained at: {run}', file=sys.stderr, flush=True)
            continue
        shutil.rmtree(work)
        print(f'Complete: {destination / "index.html"}', flush=True)
    if failures:
        raise SystemExit('Incomplete models: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
