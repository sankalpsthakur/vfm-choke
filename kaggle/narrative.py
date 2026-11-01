"""Build a narrative Kaggle notebook from a percent-format script: `# %%` starts a code cell, `# %% [markdown]` a
markdown cell whose lines carry a leading `# `.

usage: python3 kaggle/narrative.py <source.py> <slug> "<Title>" [--internet] [--dataset owner/slug]...

Writes kaggle/<slug>/<slug>.ipynb and kernel-metadata.json from the same allowlist as build.py: a fresh id under
sankalpsthakur, is_private true, CPU, no id_no (Oct 28 incident rule). Going public is a separate, explicit step.
"""
import argparse
import json
from pathlib import Path

from build import OWNER, ROOT, code_cell, md_cell, slugify


def cells_from(source):
    cells, kind, buf = [], None, []

    def flush():
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        if kind == 'markdown':
            cells.append(md_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in buf) + '\n'))
        elif kind == 'code' and buf:
            cells.append(code_cell('\n'.join(buf) + '\n'))

    for line in source.splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('markdown' if '[markdown]' in line else 'code'), []
        elif kind:
            buf.append(line)
    flush()
    return cells


def build(src, slug, title, internet=False, datasets=()):
    assert slugify(title) == slug, f'title {title!r} slugifies to {slugify(title)!r}, not {slug!r}'
    nb = {'cells': cells_from(Path(src).read_text()),
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 4}
    meta = {'id': f'{OWNER}/{slug}', 'title': title, 'code_file': f'{slug}.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': bool(internet), 'dataset_sources': list(datasets), 'competition_sources': [],
            'kernel_sources': [], 'model_sources': [], 'keywords': []}
    assert 'id_no' not in meta
    out = ROOT / 'kaggle' / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1))
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    print(f'built {out} ({len(nb["cells"])} cells, internet={bool(internet)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('slug')
    ap.add_argument('title')
    ap.add_argument('--internet', action='store_true')
    ap.add_argument('--dataset', action='append', default=[])
    a = ap.parse_args()
    build(a.src, a.slug, a.title, a.internet, a.dataset)
