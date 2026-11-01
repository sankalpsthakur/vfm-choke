# B2 probe: download the Dabiri et al. supplementary DOCX, report its table layout, keep the raw tables as outputs.
import json
import sys

sys.path.insert(0, '.')
from benchmarks import dabiri  # noqa: E402

data, url = dabiri.download()
print('downloaded', len(data), 'bytes from', url)
with open('dabiri_supplementary.docx', 'wb') as fh:
    fh.write(data)
tabs = dabiri.tables(data)
report = []
for i, rows in enumerate(tabs):
    print(f'--- table {i}: {len(rows)} rows; widths {sorted(set(len(r) for r in rows))}')
    for r in rows[:4]:
        print('   ', r)
    report.append({'table': i, 'rows': len(rows), 'head': rows[:4]})
    import csv
    with open(f'dabiri_table_{i}.csv', 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
with open('dabiri_probe.json', 'w') as fh:
    json.dump({'url': url, 'bytes': len(data), 'tables': report}, fh, indent=1)
