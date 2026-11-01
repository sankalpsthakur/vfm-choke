"""B1: Equinor Volve daily production data (Equinor Open Data Licence), loaded from the Kaggle copy.

Per well and day: on-stream hours, wellhead pressure and temperature, choke opening (%), pressure drop over the
choke, and allocated oil, gas and water volumes. Rates are volumes scaled to a full day.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCERS = ['NO 15/9-F-1 C', 'NO 15/9-F-11 H', 'NO 15/9-F-12 H', 'NO 15/9-F-14 H', 'NO 15/9-F-15 D']
FOUR = ['NO 15/9-F-11 H', 'NO 15/9-F-12 H', 'NO 15/9-F-14 H', 'NO 15/9-F-15 D']


def find_xlsx(root='/kaggle/input'):
    hits = sorted(Path(root).rglob('*olve*roduction*.xlsx'))
    if not hits:
        raise FileNotFoundError('Volve production workbook not found under ' + root)
    return hits[0]


def load(path=None, min_hours=20.0):
    path = path or find_xlsx()
    raw = pd.read_excel(path, sheet_name='Daily Production Data')
    raw = raw[raw.WELL_BORE_CODE.isin(PRODUCERS) & (raw.FLOW_KIND == 'production')].copy()
    raw['date'] = pd.to_datetime(raw.DATEPRD)
    hrs = raw.ON_STREAM_HRS.astype(float)
    full = 24.0 / hrs.where(hrs > 0)
    df = pd.DataFrame({
        'well': raw.WELL_BORE_CODE, 'date': raw.date, 'hours': hrs,
        'opening': raw.AVG_CHOKE_SIZE_P.astype(float) / 100.0,
        'p_wh_barg': raw.AVG_WHP_P.astype(float), 't_wh_c': raw.AVG_WHT_P.astype(float),
        'dp_choke_bar': raw.DP_CHOKE_SIZE.astype(float),
        'q_o': raw.BORE_OIL_VOL.astype(float) * full, 'q_g': raw.BORE_GAS_VOL.astype(float) * full,
        'q_w': raw.BORE_WAT_VOL.astype(float) * full,
    })
    ok = ((df.hours >= min_hours) & (df.q_o > 0) & (df.opening > 0.01) & (df.p_wh_barg > 1)
          & (df.dp_choke_bar > 0) & (df.dp_choke_bar < df.p_wh_barg) & df.t_wh_c.between(1, 150))
    df = df[ok].sort_values(['well', 'date']).reset_index(drop=True)
    df['day'] = (df.date - df.groupby('well').date.transform('min')).dt.days
    return df
