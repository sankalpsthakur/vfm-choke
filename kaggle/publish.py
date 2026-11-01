"""Publish a kernel that has already run privately: push it once more with is_private false and its tags, wait for
the run, fetch its outputs, and confirm the live kernel is public.

usage: python3 kaggle/publish.py <slug> --tag energy --tag regression ...

Publishing needs Sankalp's explicit go, logged in Documentation.md first. Guards (Oct 28 incident rules): the
metadata must name sankalpsthakur/<slug> and carry no id_no; the live kernel must already exist under that id; the
push reply must name the slug. The local metadata keeps is_private false afterwards, so push.py (private only) will
refuse this kernel from now on instead of flipping it back.
"""
import argparse
import json
import re
import shutil
import sys
import time

from push import OWNER, ROOT, log, run, status


def live_meta(ref, slug):
    probe = ROOT / 'kaggle' / '.probe' / slug
    shutil.rmtree(probe, ignore_errors=True)
    run(['kaggle', 'kernels', 'pull', ref, '-p', str(probe), '-m'], 120)
    return json.loads((probe / 'kernel-metadata.json').read_text())


def main(slug, tags):
    assert 1 <= len(tags) <= 5, 'Kaggle keeps at most 5 tags, and tags cannot be removed once added'
    d = ROOT / 'kaggle' / slug
    meta = json.loads((d / 'kernel-metadata.json').read_text())
    ref = f'{OWNER}/{slug}'
    assert meta['id'] == ref and 'id_no' not in meta, 'metadata failed the allowlist'
    assert live_meta(ref, slug)['id'] == ref, f'{ref} does not exist yet: run it privately with push.py first'
    while status(ref) in ('RUNNING', 'QUEUED'):
        log(f'{ref} is still running; waiting')
        time.sleep(60)
    meta['is_private'] = False
    meta['keywords'] = list(tags)
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
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
    if 'not valid' in out:
        log(f'tag warning: {out[-300:]}')
    ver = re.search(r'version (\d+)', out)
    v = ver.group(1) if ver else 'x'
    log(f'pushed {ref} v{v} as public')
    t0 = time.time()
    time.sleep(30)
    while (s := status(ref)) in ('RUNNING', 'QUEUED'):
        time.sleep(30)
    log(f'{ref} v{v} finished: {s} after {time.time() - t0:.0f} s')
    dest = ROOT / 'reports' / 'runs' / f'{slug}-v{v}'
    dest.mkdir(parents=True, exist_ok=True)
    run(['kaggle', 'kernels', 'output', ref, '-p', str(dest), '-o'], 900)
    (dest / 'log.raw').write_text(run(['kaggle', 'kernels', 'logs', ref], 300))
    live = live_meta(ref, slug)
    (dest / 'status.json').write_text(json.dumps({'ref': ref, 'version': v, 'status': s,
                                                  'is_private': live.get('is_private'),
                                                  'keywords': live.get('keywords')}, indent=1))
    log(f'outputs in {dest}; live is_private={live.get("is_private")} keywords={live.get("keywords")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('slug')
    ap.add_argument('--tag', action='append', default=[])
    a = ap.parse_args()
    main(a.slug, a.tag)
