"""Download and verify a pinned public RoboTwin post-trained checkpoint."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = 'robbyant/lingbot-vla-v2-6b-robotwin'
REVISION = '0451855729ec904f970600e0aec8b84661423afe'
ROOT = Path(__file__).resolve().parents[1] / 'models/lingbot-vla-v2-6b-robotwin'


def main():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {'repo': REPO, 'revision': revision, 'files': metadata['siblings']}
    path = ROOT / 'download_manifest.json'
    if path.exists() and json.loads(path.read_text())['revision'] != revision:
        raise RuntimeError('Existing model directory contains another revision')
    path.write_text(json.dumps(manifest, indent=2) + '\n')

    def verified(file, entry):
        if not file.is_file() or file.stat().st_size != entry['size']:
            return False
        lfs = entry.get('lfs')
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f"blob {entry['size']}\0".encode())
        with file.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        expected = lfs['sha256'] if lfs else entry['blobId']
        return digest.hexdigest() == expected

    def download(entry):
        relative = Path(entry['rfilename'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(relative)
        final = ROOT / relative
        final.parent.mkdir(parents=True, exist_ok=True)
        if verified(final, entry):
            return {'file': str(relative), 'bytes': entry['size'], 'verified': True}
        partial = final.with_name(final.name + '.partial')
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= entry['size']:
                    if verified(partial, entry):
                        partial.replace(final)
                        break
                    partial.unlink(); offset = 0
                url = f'https://huggingface.co/{REPO}/resolve/{revision}/{relative.as_posix()}?download=true'
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected partial response range')
                    with partial.open('ab' if append else 'wb') as stream:
                        while chunk := response.read(8 * 1024 * 1024):
                            stream.write(chunk)
                if not verified(partial, entry):
                    raise ValueError(f'Checksum/size mismatch: {relative}')
                partial.replace(final)
                break
            except Exception as error:
                print('RETRY', str(relative), attempt + 1, type(error).__name__, flush=True)
                if attempt == 4:
                    raise
                time.sleep(3)
        return {'file': str(relative), 'bytes': entry['size'], 'verified': True}

    entries = metadata['siblings']
    small = [x for x in entries if x['size'] < 20_000_000]
    large = [x for x in entries if x['size'] >= 20_000_000]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for entry in small:
            results.append(download(entry))
        print('CONFIG_READY', str(ROOT/'lingbotvla_cli.yaml'), revision, flush=True)
        pending = {pool.submit(download, entry): entry for entry in large}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result(); results.append(result)
                pending.pop(future)
                print('VERIFIED', result['file'], result['bytes'], flush=True)
            total = sum((ROOT/x['rfilename']).stat().st_size if (ROOT/x['rfilename']).exists() else
                        ((ROOT/(x['rfilename']+'.partial')).stat().st_size if (ROOT/(x['rfilename']+'.partial')).exists() else 0)
                        for x in entries)
            print('PROGRESS', total, '/', sum(x['size'] for x in entries), flush=True)
    report = {'repo': REPO, 'revision': revision, 'verified': True, 'total_bytes': sum(x['bytes'] for x in results), 'files': results}
    (ROOT/'download_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DOWNLOAD_COMPLETE', json.dumps({'revision': revision, 'bytes': report['total_bytes'], 'files': len(results)}), flush=True)


if __name__ == '__main__':
    main()
