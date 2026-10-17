"""Render a standalone, offline results viewer from results.csv."""
import csv
import html
import json
from pathlib import Path


def render(root):
    root = Path(root).resolve()
    with (root / 'results.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No results in CSV')
    for row in rows:
        row['success'] = int(row['success'])
        if row['success'] not in (0, 1):
            raise ValueError('Invalid success value')
        video = (root / row['video']).resolve()
        if not video.is_relative_to(root / 'videos') or not video.is_file():
            raise ValueError(f"Missing or invalid video: {row['video']}")
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    model_label = rows[0].get('model') or 'LingBot-VLA-v2-6B RoboTwin 后训练版'
    # Legacy CSV files were clean-only and have no setting column.
    settings = {row.get('setting') or 'demo_clean' for row in rows}
    labels = {'demo_clean': 'Clean', 'demo_randomized': 'Randomized'}
    setting_label = ' / '.join(labels.get(value, value) for value in sorted(settings))
    page = (TEMPLATE.replace('__MODEL__', html.escape(model_label))
            .replace('__SETTING__', html.escape(setting_label)).replace('__RESULTS__', payload))
    (root / 'index.html').write_text(page, encoding='utf-8')
    return rows


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTwin 测试结果</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#172033;background:#f4f6fa;color-scheme:light}*{box-sizing:border-box}body{margin:0}main{max-width:1380px;margin:auto;padding:36px 28px}h1{font-size:30px;letter-spacing:-.8px;margin:4px 0 10px}.eyebrow{color:#5365c5;font-size:12px;letter-spacing:2px;font-weight:750}.muted{color:#657187;font-size:14px;line-height:1.7}.top{display:flex;justify-content:space-between;align-items:center;gap:20px}.download{color:#3547aa;background:white;border:1px solid #dce1ed;border-radius:10px;padding:11px 16px;text-decoration:none;white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.stat{background:white;border:1px solid #e1e6ef;border-radius:14px;padding:18px 22px}.stat strong{display:block;font-size:32px;margin-bottom:4px}.stat span{color:#657187;font-size:13px}.good{color:#158261}.bad{color:#c04c48}.layout{display:grid;grid-template-columns:minmax(340px,.9fr) minmax(420px,1.35fr);gap:20px;align-items:start}.panel{background:#fff;border:1px solid #e1e6ef;border-radius:16px;overflow:hidden}.toolbar{padding:18px;border-bottom:1px solid #edf0f6}input{width:100%;padding:12px;border:1px solid #d9dfeb;border-radius:9px;font:inherit;font-size:14px}input:focus-visible,button:focus-visible,a:focus-visible{outline:3px solid #8296ef;outline-offset:2px}.filters{display:flex;gap:8px;margin-top:12px}button{font:inherit;cursor:pointer}.filter{padding:7px 13px;background:#f4f6fa;border:1px solid transparent;border-radius:8px;color:#5b667a;font-size:13px}.filter[aria-pressed=true]{background:#eaf0ff;color:#314ba8;border-color:#c8d4f7}.count{font-size:12px;color:#7a8598;margin:12px 0 0}.task-list{max-height:590px;overflow:auto;padding:8px}.task{width:100%;display:flex;align-items:center;gap:12px;text-align:left;padding:14px 12px;border:0;border-radius:9px;background:white;margin:2px 0}.task:hover{background:#f5f7fc}.task[aria-pressed=true]{background:#edf2ff;box-shadow:inset 3px 0 #586fd3}.name{font-size:13px;overflow-wrap:anywhere;flex:1}.badge{font-size:11px;font-weight:650;padding:4px 8px;border-radius:6px;white-space:nowrap}.badge.good{background:#e7f5ee}.badge.bad{background:#fceceb}.empty{padding:40px;color:#657187;text-align:center}.viewer{padding:22px;position:sticky;top:20px}h2{font-size:20px;overflow-wrap:anywhere;margin:0 0 14px}video{display:block;width:100%;aspect-ratio:4/3;max-height:480px;background:#121620;border-radius:10px;object-fit:contain}.details{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.detail{background:#f6f8fc;border-radius:9px;padding:11px 12px}.detail span{display:block;font-size:12px;color:#758097;margin-bottom:4px}.detail strong{font-size:15px}.instruction{font-size:14px;line-height:1.6;min-height:44px;margin:8px 0 12px}.video-link{font-size:13px;color:#425dbb}.footer{margin-top:22px;font-size:12px;color:#7b8698;line-height:1.8}@media(max-width:860px){main{padding:24px 16px}.layout{grid-template-columns:1fr}.viewer{position:static;grid-row:1}.task-list{max-height:380px}.stat{padding:14px}.stat strong{font-size:27px}.top{align-items:start}h1{font-size:25px}}@media(max-width:480px){.stats{grid-template-columns:repeat(2,1fr)}.download{font-size:12px;padding:9px}.details{gap:6px}.viewer{padding:15px}}
</style>
</head>
<body><main>
<div class="top"><div><div class="eyebrow">ROBOTWIN · EVALUATION</div><h1>RoboTwin 测试结果</h1><div class="muted">__MODEL__<br>__SETTING__ / unseen · 每任务 1 回合</div></div><a class="download" href="results.csv" download>下载 CSV</a></div>
<div class="stats"><div class="stat"><strong id="rate">—</strong><span>成功率</span></div><div class="stat"><strong id="total">—</strong><span>测试任务</span></div><div class="stat"><strong class="good" id="success">—</strong><span>成功</span></div><div class="stat"><strong class="bad" id="failure">—</strong><span>失败</span></div></div>
<div class="layout"><section class="panel" aria-label="任务列表"><div class="toolbar"><input id="search" type="search" placeholder="搜索任务名或指令" aria-label="搜索任务名或指令"><div class="filters"><button class="filter" data-filter="all" aria-pressed="true">全部</button><button class="filter" data-filter="failed" aria-pressed="false">仅失败</button><button class="filter" data-filter="success" aria-pressed="false">仅成功</button></div><p id="count" class="count" aria-live="polite"></p></div><div id="tasks" class="task-list"></div></section>
<section class="panel viewer" aria-label="任务视频"><h2 id="title"></h2><video id="video" controls playsinline preload="metadata" aria-label="任务回合视频"></video><div class="details"><div class="detail"><span>结果</span><strong id="status"></strong></div><div class="detail"><span>场景种子</span><strong id="seed"></strong></div><div class="detail"><span>模型调用</span><strong id="calls"></strong></div></div><div class="muted">任务指令</div><p id="instruction" class="instruction"></p><a id="video-link" class="video-link" target="_blank" rel="noopener">单独打开视频 ↗</a><p id="video-error" class="muted" hidden>浏览器未能播放视频，请使用上面的链接，或用本地播放器打开对应 MP4。</p></section></div>
<p class="footer">FP32 权重，未量化。本次为本地适配运行，每任务仅 1 回合，非官方完整基准复现。<br>视频和 CSV 均在当前目录，页面无需联网。分享时请保留 index.html、results.csv 与 videos 文件夹的相对位置。</p>
</main><script id="results-data" type="application/json">__RESULTS__</script><script>
'use strict';
const rows = JSON.parse(document.getElementById('results-data').textContent);
const byId = id => document.getElementById(id);
const passed = rows.reduce((sum, row) => sum + row.success, 0);
byId('rate').textContent = `${(100 * passed / rows.length).toFixed(0)}%`;
byId('total').textContent = rows.length; byId('success').textContent = passed; byId('failure').textContent = rows.length - passed;
let filter = 'all', selected = null;
function select(row) {
  selected = row.task;
  byId('title').textContent = row.task;
  byId('status').textContent = row.success ? '成功' : '失败';
  byId('status').className = row.success ? 'good' : 'bad';
  byId('seed').textContent = row.episode_seed; byId('calls').textContent = row.policy_calls;
  byId('instruction').textContent = row.instruction;
  byId('video-error').hidden = true;
  byId('video').pause(); byId('video').src = row.video; byId('video').load();
  byId('video-link').href = row.video;
  document.querySelectorAll('.task').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.task === selected)));
}
function renderList() {
  const query = byId('search').value.trim().toLowerCase();
  const visible = rows.filter(row => (filter === 'all' || (filter === 'failed' ? !row.success : row.success)) && `${row.task} ${row.instruction}`.toLowerCase().includes(query)).sort((a,b) => a.success-b.success || a.task.localeCompare(b.task));
  byId('tasks').replaceChildren(); byId('count').textContent = `显示 ${visible.length} / ${rows.length} 个任务 · 失败优先`;
  if (!visible.length) { const empty = document.createElement('p'); empty.className='empty'; empty.textContent='没有匹配的任务'; byId('tasks').append(empty); return; }
  visible.forEach(row => {
    const button = document.createElement('button'); button.className='task'; button.dataset.task=row.task; button.setAttribute('aria-pressed', String(row.task===selected));
    const name = document.createElement('span'); name.className='name'; name.textContent=row.task;
    const badge = document.createElement('span'); badge.className=`badge ${row.success ? 'good' : 'bad'}`; badge.textContent=row.success ? '成功' : '失败';
    button.append(name,badge); button.addEventListener('click', () => select(row)); byId('tasks').append(button);
  });
  if (!visible.some(row => row.task===selected)) select(visible[0]);
}
document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {filter=button.dataset.filter; document.querySelectorAll('.filter').forEach(item => item.setAttribute('aria-pressed', String(item===button)));renderList();}));
byId('search').addEventListener('input', renderList);
byId('video').addEventListener('error', () => {byId('video-error').hidden=false;});
renderList();
</script></body></html>'''
