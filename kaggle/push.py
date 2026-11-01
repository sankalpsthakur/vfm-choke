"""Push a built kernel privately, wait for it to finish, and fetch its outputs and log right away.

usage: python3 kaggle/push.py <slug>

Guards (Oct 28 incident rules): the metadata must be private, carry no id_no, and name sankalpsthakur/<slug>; an
existing kernel under that slug must already be private; the push reply must name the slug, otherwise stop.
Outputs land in reports/runs/<slug>-v<version>/ (versioned refs return the latest run, so fetch before re-pushing).
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'sankalpsthakur'
NOISE = ('Warning: Looks like', ' please consider')


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return '\n'.join(l for l in (p.stdout + p.stderr).splitlines() if not l.startswith(NOISE))


def status(ref):
    out = run(['kaggle', 'kernels', 'status', ref], 90)
    return next((s for s in ('RUNNING', 'QUEUED', 'COMPLETE', 'ERROR', 'CANCEL') if s in out), out[-200:])


def main(slug):
    d = ROOT / 'kaggle' / slug
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    ref = f'{OWNER}/{slug}'
    assert meta['id'] == ref and meta['is_private'] is True and 'id_no' not in meta, 'metadata failed the allowlist'
    probe = ROOT / 'kaggle' / '.probe' / slug
    shutil.rmtree(probe, ignore_errors=True)
    run(['kaggle', 'kernels', 'pull', ref, '-p', str(probe), '-m'], 120)
    if (probe / 'kernel-metadata.json').exists():
        live = json.loads((probe / 'kernel-metadata.json').read_text())
        assert live.get('is_private') is True, f'{ref} exists and is not private: STOP'
        while status(ref) in ('RUNNING', 'QUEUED'):
            log(f'{ref} is still running; waiting')
            time.sleep(60)
    out = ''
    for attempt in range(48):
        out = run(['kaggle', 'kernels', 'push', '-p', str(d)], 300)
        if 'Maximum batch' not in out:
            break
        log(f'session cap full (attempt {attempt + 1}); retrying in 5 min')
        time.sleep(300)
    m = re.search(r'kaggle\.com/(?:code/)?' + OWNER + r'/([a-z0-9-]+)', out)
    if not m or m.group(1) != slug:
        sys.exit(f'STOP: push reply does not name {slug}: {out[-400:]}')
    ver = re.search(r'version (\d+)', out)
    v = ver.group(1) if ver else 'x'
    log(f'pushed {ref} v{v}')
    t0 = time.time()
    time.sleep(30)
    while (s := status(ref)) in ('RUNNING', 'QUEUED'):
        time.sleep(30)
    log(f'{ref} v{v} finished: {s} after {time.time() - t0:.0f} s')
    dest = ROOT / 'reports' / 'runs' / f'{slug}-v{v}'
    dest.mkdir(parents=True, exist_ok=True)
    run(['kaggle', 'kernels', 'output', ref, '-p', str(dest), '-o'], 900)
    (dest / 'log.raw').write_text(run(['kaggle', 'kernels', 'logs', ref], 300))
    (dest / 'status.json').write_text(json.dumps({'ref': ref, 'version': v, 'status': s}, indent=1))
    log(f'outputs in {dest}')


if __name__ == '__main__':
    main(sys.argv[1])
