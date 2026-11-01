"""B2: Dabiri et al. 2024 (Sci Rep 14:6945), 565 wellhead-choke tests from eight literature sources.

Inputs Pwh (psi), choke size Dc (64ths of an inch), GLR (scf/stb); target liquid rate (stb/d). The data ships as a
DOCX supplementary file (CC BY 4.0); this module downloads and parses it without third-party DOCX libraries.
"""
import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

URLS = [
    'https://pmc.ncbi.nlm.nih.gov/articles/instance/10960849/bin/41598_2024_54010_MOESM1_ESM.docx',
    'https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-54010-2/MediaObjects/'
    '41598_2024_54010_MOESM1_ESM.docx',
]
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def download():
    import urllib.request
    last = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research benchmark)'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if data[:2] == b'PK':
                return data, url
            last = f'{url}: not a DOCX ({data[:40]!r})'
        except Exception as e:  # try the next mirror
            last = f'{url}: {e}'
    raise RuntimeError(f'could not download the Dabiri supplementary file; last error {last}')


def tables(docx_bytes):
    """All tables in the document as lists of rows of cell strings."""
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(docx_bytes)).read('word/document.xml'))
    out = []
    for tbl in root.iter(W + 'tbl'):
        rows = []
        for tr in tbl.iter(W + 'tr'):
            rows.append([''.join(t.text or '' for t in tc.iter(W + 't')).strip() for tc in tr.iter(W + 'tc')])
        out.append(rows)
    return out


def _num(s):
    s = s.replace(',', '').replace('−', '-').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse(docx_bytes):
    """Rows with four numeric values in the order the paper uses (Pwh, Dc, GLR, Qliq), plus any text label that
    looks like a source reference. Column order is checked against the paper's ranges."""
    recs, headers = [], []
    for ti, rows in enumerate(tables(docx_bytes)):
        for row in rows:
            nums = [_num(c) for c in row]
            vals = [v for v in nums if np.isfinite(v)]
            text = [c for c, v in zip(row, nums) if not np.isfinite(v) and c]
            if len(vals) >= 4:
                recs.append({'table': ti, 'vals': vals, 'text': ' '.join(text)})
            elif text:
                headers.append((ti, row))
    if not recs:
        raise RuntimeError('no numeric rows found in the DOCX tables')
    n_cols = pd.Series([len(r['vals']) for r in recs]).mode().iloc[0]
    arr = np.array([r['vals'][-n_cols:] for r in recs if len(r['vals']) >= n_cols])
    return arr, recs, headers


def assign_columns(arr):
    """Pick the columns that match the paper's ranges: Pwh 50-4308 psi, Dc 14-80 /64 in, GLR 36-13095 scf/stb,
    Qliq 198-25878 stb/d. Returns a DataFrame and the mapping used."""
    ranges = {'pwh': (40, 5000), 'd64': (10, 90), 'glr': (30, 14000), 'qliq': (150, 30000)}
    cols = list(range(arr.shape[1]))
    best, best_score = None, -1
    from itertools import permutations
    for perm in permutations(cols, 4):
        score = 0
        for name, c in zip(ranges, perm):
            lo, hi = ranges[name]
            score += np.mean((arr[:, c] >= lo) & (arr[:, c] <= hi))
        # Dc is the only small-integer-like column
        score += np.mean(np.abs(arr[:, perm[1]] - np.round(arr[:, perm[1]])) < 1e-6) * 0.5
        if score > best_score:
            best, best_score = perm, score
    df = pd.DataFrame({name: arr[:, c] for name, c in zip(ranges, best)})
    return df, dict(zip(ranges, best)), best_score
