"""Download and validate only RoboTwin simulator assets, then extract them."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import stat
import urllib.request
import zipfile

spec = importlib.util.spec_from_file_location('clean_downloader', str(Path(__file__).with_name('download_robotwin_clean50.py')))
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
root = Path('/srv/common_dataset/RoboTwin2.0')
downloader.ROOT = root / 'asset_archives'
downloader.ROOT.mkdir(parents=True, exist_ok=True)
assets = root / 'assets'
assets.mkdir(exist_ok=True)
url = 'https://huggingface.co/api/datasets/' + downloader.REPO + '/revision/' + downloader.REVISION + '?blobs=true'
with urllib.request.urlopen(url, timeout=60) as response:
    metadata = json.load(response)
names = {'objects.zip', 'embodiments.zip', 'background_texture.zip'}
entries = [x for x in metadata['siblings'] if x['rfilename'] in names]
assert len(entries) == 3
(root / 'asset_manifest.json').write_text(json.dumps({'revision': downloader.REVISION, 'files': entries}, indent=2))

def prepare(entry):
    name = entry['rfilename']
    print('DOWNLOADING', name, entry['size'], flush=True)
    result = downloader.download(entry)
    print('VERIFIED', name, flush=True)
    marker = assets / ('.' + name + '.complete')
    if marker.exists() and marker.read_text().strip() == entry['lfs']['sha256']:
        return result
    print('EXTRACTING', name, flush=True)
    with zipfile.ZipFile(downloader.ROOT / name) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('unsafe archive member: ' + info.filename)
        archive.extractall(assets)
    marker.write_text(entry['lfs']['sha256'] + '\n')
    print('READY', name, flush=True)
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(prepare, entries))
(root / 'asset_verification.json').write_text(json.dumps(results, indent=2))
print('COMPLETE: all three simulator asset archives verified and extracted.', flush=True)
