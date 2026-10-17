"""Validate and summarize a completed multi-worker RoboTwin evaluation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if (root/'videos').is_dir() and (root/'results.csv').is_file():
        from evaluation.render_results import render
        rows = render(root)
        print(json.dumps({'evaluated_tasks': len(rows), 'successes': sum(r['success'] for r in rows),
                          'html': str(root/'index.html')}, indent=2))
        return
    organized = (root/'run_metadata.json').is_file()
    if organized:
        plan = json.loads((root/'run_metadata.json').read_text())
        workers = [{'output': str(root/'cases'), 'tasks': plan['tasks']}]
        checks = [{'worker': run['batch_id'], 'model': run['model_validation']}
                  for run in plan['execution_batches']]
    else:
        plan = json.loads((root/'run_plan.json').read_text())
        workers = plan['workers']
        checks = [{'worker': worker['index'],
                   'model': json.loads((Path(worker['output'])/'model_ready.json').read_text())}
                  for worker in workers]
    for check in checks:
        ready = check['model']
        assert ready['precision_mode'] == 'fp32'
        assert set(ready['parameter_dtypes']) == {'torch.float32'}
        assert ready['chunk_size'] == 50 and ready['denoising_steps'] == 10
    records = {}
    for worker in workers:
        directory = Path(worker['output'])
        for task in worker['tasks']:
            result_path = directory/task/'result.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            if result['status'] != 'complete' or result['episodes'] != 1:
                continue
            assert result['setting'] == 'demo_clean' and result['instruction_type'] == 'unseen'
            assert len(result['requests']) == 1
            request = result['requests'][0]
            assert request['policy_calls'] > 0
            assert len(request['action_shapes']) == request['policy_calls']
            assert all(s == [50,14] for s in request['action_shapes'])
            actions = np.load(directory/task/'first_actions_000.npy')
            assert actions.shape == (50,14) and np.isfinite(actions).all()
            videos = list((directory/task).glob('*.mp4'))
            assert len(videos) == 1 and videos[0].stat().st_size > 1000
            records[task] = {'task':task, 'success':int(result['successes']),
                'policy_calls':request['policy_calls'], 'elapsed_seconds':round(result['elapsed_seconds'],2),
                'episode_seed':result['final_seed']-1, 'instruction':request['instruction'],
                'result':str(result_path), 'video':str(videos[0])}
    rows = [records[t] for t in plan['tasks'] if t in records]
    missing = [t for t in plan['tasks'] if t not in records]
    successes = sum(row['success'] for row in rows)
    summary = {'complete':not missing, 'model_repo':plan['model_repo'], 'model_revision':plan['model_revision'],
        'setting':'demo_clean', 'instruction_type':'unseen', 'precision':'official fp32',
        'expected_tasks':len(plan['tasks']), 'evaluated_tasks':len(rows), 'episodes_per_task':1,
        'successes':successes, 'success_rate':successes/len(rows) if rows else None,
        'failed_tasks':[r['task'] for r in rows if not r['success']], 'missing_tasks':missing,
        'validation':checks, 'tasks':rows,
        'scope':'One episode per task; not the published 100-episode-per-task benchmark. No randomized evaluation in this run.'}
    if not organized:
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if rows:
        with (root/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# RoboTwin 官方后训练模型实测','',f"模型：{plan['model_repo']}，版本 `{plan['model_revision']}`。",
        '',f'Clean / unseen / 官方 FP32；每任务 1 回合。已完成 {len(rows)}/{len(plan["tasks"])} 个任务，成功 {successes} 个。',
        '',f'实测成功率：{successes/len(rows):.1%}。' if rows else '尚无有效回合。',
        '', '这是 50 回合快速测试，不能视为官方完整基准复现。', '', '| 任务 | 成功 | 模型调用次数 | 用时（秒） |','|---|---:|---:|---:|']
    lines += [f"| {r['task']} | {r['success']} | {r['policy_calls']} | {r['elapsed_seconds']} |" for r in rows]
    if missing:lines+=['','未完成：'+', '.join(missing)]
    if organized:
        lines = ['# RoboTwin 后训练模型测试结果', '',
            f'**{len(rows)} 个任务：成功 {successes} 个，失败 {len(rows)-successes} 个，成功率 {successes/len(rows):.0%}。**', '',
            '模型：`robbyant/lingbot-vla-v2-6b-robotwin`（step 50000）。Clean / unseen，每任务 1 回合，FP32 权重、无量化。',
            '本次是本地适配评测，未完全复现官方运行环境。', '',
            '直接在下面点击视频观看，失败任务排在前面。全部文件按任务名放在 `cases/`，不再按运行批次划分。', '',
            '- `results.csv`：50 个任务的成绩、指令、种子、用时及文件路径。',
            '- `cases/<任务名>/video.mp4`：完整回合视频。',
            '- 同目录的 `result.json`、`evaluation.log`：结果与执行日志。',
            '- 同目录的 `observation_000.npz`、`first_actions_000.npy`：初始观测和首个动作块。',
            '- `run_metadata.json`：模型版本、运行配置和校验信息；仅在复现或排错时需要查看。', '',
            '| 任务 | 结果 | 视频 |', '|---|---|---|']
        for row in sorted(rows, key=lambda r: (r['success'], r['task'])):
            task = row['task']; status = '成功' if row['success'] else '**失败**'
            lines.append(f"| [{task}]({root/'cases'/task}) | {status} | [播放]({row['video']}) |")
        (root/'README.md').write_text('\n'.join(lines)+'\n')
    else:
        (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['complete','evaluated_tasks','successes','success_rate','failed_tasks','missing_tasks']},indent=2))


if __name__ == '__main__':
    main()
