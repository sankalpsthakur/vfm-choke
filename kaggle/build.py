"""Build a self-contained Kaggle notebook: the package (and tests) as %%writefile cells, then an entry script.

usage: python3 kaggle/build.py <entry> <slug> "<Title>" [--internet] [--tests] [--dataset owner/slug]...
       [--kernel owner/slug]...

<entry> is a file in kaggle/entries/. Writes kaggle/<slug>/<slug>.ipynb and kernel-metadata.json. The metadata is
built from an allowlist only: a fresh id under sankalpsthakur, is_private true, CPU, no id_no (Oct 28 incident rule).
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = 'sankalpsthakur'


def slugify(title):
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': src.splitlines(keepends=True)}


def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(keepends=True)}


def build(entry, slug, title, internet=False, tests=False, datasets=(), kernels=()):
    assert slugify(title) == slug, f'title {title!r} slugifies to {slugify(title)!r}, not {slug!r}'
    files = sorted((ROOT / 'src' / 'vfmchoke').glob('*.py'))
    if tests:
        files += sorted((ROOT / 'tests').glob('*.py'))
    for extra in ('benchmarks',):
        files += sorted((ROOT / extra).glob('*.py'))
    dirs = sorted({str(f.parent.relative_to(ROOT)) for f in files})
    cells = [md_cell(f'# {title}\n\nPrivate run of `vfmchoke` ({entry}). The package source is written by the cells '
                     'below, so this notebook is exactly the code that ran.\n'),
             code_cell('import os\n' + ''.join(f'os.makedirs({d!r}, exist_ok=True)\n' for d in dirs))]
    for f in files:
        body = f.read_text() or '# package marker\n'          # %%writefile refuses an empty cell body
        cells.append(code_cell(f'%%writefile {f.relative_to(ROOT)}\n' + body))
    cells.append(code_cell((ROOT / 'kaggle' / 'entries' / entry).read_text()))
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                      'name': 'python3'},
                                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 4}
    meta = {'id': f'{OWNER}/{slug}', 'title': title, 'code_file': f'{slug}.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': bool(internet), 'dataset_sources': list(datasets), 'competition_sources': [],
            'kernel_sources': list(kernels), 'model_sources': [], 'keywords': []}
    assert 'id_no' not in meta
    out = ROOT / 'kaggle' / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1))
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    print(f'built {out} ({len(files)} files, internet={bool(internet)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('entry')
    ap.add_argument('slug')
    ap.add_argument('title')
    ap.add_argument('--internet', action='store_true')
    ap.add_argument('--tests', action='store_true')
    ap.add_argument('--dataset', action='append', default=[])
    ap.add_argument('--kernel', action='append', default=[])
    a = ap.parse_args()
    build(a.entry, a.slug, a.title, a.internet, a.tests, a.dataset, a.kernel)
