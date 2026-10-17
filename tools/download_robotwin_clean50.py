#!/usr/bin/env python3
"""Download only the 50 competition clean archives using Python's standard library."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

ROOT = Path('/srv/common_dataset/RoboTwin2.0')
REPO = 'TianxingChen/RoboTwin2.0'
REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'

def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60)

def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError('file size mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != entry['lfs']['sha256']:
        raise ValueError('SHA256 mismatch')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        episodes = sum(x.filename.endswith(('.hdf5', '.h5')) for x in members)
        unpacked = sum(x.file_size for x in members)
    return {'file': entry['rfilename'], 'bytes': entry['size'],
            'sha256_verified': True, 'hdf5_episode_count': episodes,
            'uncompressed_bytes': unpacked}

def download(entry):
    destination = ROOT / entry['rfilename']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verify(destination, entry)
    partial = destination.with_suffix('.zip.part')
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry['size']:
                raise ValueError('partial file is larger than expected; inspect it before retrying')
            if offset < entry['size']:
                url = ('https://huggingface.co/datasets/' + REPO + '/resolve/' + REVISION
                       + '/' + entry['rfilename'] + '?download=true&request_time=' + str(time.time_ns()))
                headers = {'Range': 'bytes=' + str(offset) + '-'} if offset else {}
                with request(url, headers) as response:
                    status = response.status
                    if status == 206:
                        expected = 'bytes ' + str(offset) + '-'
                        if not response.headers.get('Content-Range', '').startswith(expected):
                            raise ValueError('unexpected resume offset')
                        mode = 'ab' if offset else 'wb'
                    elif status == 200:
                        mode = 'wb'
                    else:
                        raise ValueError('unexpected HTTP status: ' + str(status))
                    with partial.open(mode) as stream:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            stream.write(block)
            try:
                result = verify(partial, entry)
            except ValueError as exc:
                if str(exc) == 'SHA256 mismatch':
                    partial.rename(partial.with_name(partial.name + '.bad-' + str(time.time_ns())))
                raise
            partial.rename(destination)
            return result
        except Exception as exc:
            print('RETRY', attempt, entry['rfilename'], type(exc).__name__, str(exc), flush=True)
            if attempt == 5:
                raise
            time.sleep(5 * attempt)

def main():
    with request('https://huggingface.co/api/datasets/' + REPO + '/revision/' + REVISION + '?blobs=true') as response:
        metadata = json.load(response)
    files = [x for x in metadata['siblings'] if x['rfilename'].startswith('dataset/')
             and x['rfilename'].endswith('/aloha-agilex_clean_50.zip')]
    assert len(files) == 50, 'expected exactly 50 clean archives'
    assert sum(x['size'] for x in files) == 23780715316, 'unexpected total size'
    assert metadata['sha'] == REVISION
    print('50 clean archives, 23.78 GB. Destination:', ROOT, flush=True)
    if '--check' in sys.argv:
        print('CHECK OK: Python standard library and official metadata are accessible. No dataset downloaded.')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'clean50_download_manifest.json').write_text(json.dumps(
        {'repo_id': REPO, 'revision': REVISION, 'files': files}, indent=2))
    results = []
    failures = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(download, entry): entry['rfilename'] for entry in files}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    (ROOT / 'clean50_verification.json').write_text(json.dumps(results, indent=2))
                    print('VERIFIED', str(len(results)) + '/50', name,
                          'episodes=' + str(result['hdf5_episode_count']), flush=True)
                except Exception as exc:
                    failures.append(name)
                    print('FAILED', name, type(exc).__name__, str(exc), flush=True)
            current_bytes = 0
            for entry in files:
                path = ROOT / entry['rfilename']
                for candidate in (path, path.with_suffix('.zip.part')):
                    try:
                        current_bytes += candidate.stat().st_size
                        break
                    except FileNotFoundError:
                        pass
            print('PROGRESS %.2f/23.78 GB, verified %d/50, elapsed %.0fs' %
                  (current_bytes / 1e9, len(results), time.monotonic() - started), flush=True)
    if failures:
        raise SystemExit('Some files failed. Run the same command again to resume: ' + ', '.join(failures))
    print('COMPLETE: 50 archives downloaded and SHA256 verified; episodes=' +
          str(sum(x['hdf5_episode_count'] for x in results)), flush=True)

if __name__ == '__main__':
    main()
