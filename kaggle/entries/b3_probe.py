# B3 probe: fetch the D'Souza et al. MATLAB simulator (GitHub and DataverseNO copies), print it, try Octave.
import json
import subprocess
import urllib.request

FILES = ['oil_field_model.m', 'openloop_simulation.m', 'make_w_gc.m', 'update_states.m', 'calculate_output.m']
GH = 'https://raw.githubusercontent.com/dsouzaneville/FMH606-1-Masters-Thesis/main/Simulator/'
DV = {'oil_field_model.m': 268788, 'openloop_simulation.m': 268789, 'make_w_gc.m': 268786,
      'update_states.m': 268787, 'calculate_output.m': 268790}
report = {}
for f in FILES:
    for tag, url in (('github', GH + f), ('dataverse', f'https://dataverse.no/api/access/datafile/{DV[f]}')):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
                                        timeout=60) as r:
                txt = r.read().decode('utf-8', 'replace')
            with open(f'{tag}_{f}', 'w') as fh:
                fh.write(txt)
            report[f'{tag}/{f}'] = len(txt)
            print(f'===== {tag}/{f} ({len(txt)} chars) =====')
            print(txt)
        except Exception as e:
            report[f'{tag}/{f}'] = f'error: {e}'
            print(f'===== {tag}/{f}: {e}')
r = subprocess.run('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq octave >/dev/null '
                   '&& octave --version | head -1', shell=True, capture_output=True, text=True, timeout=1500)
report['octave'] = (r.stdout + r.stderr)[-500:]
print('octave install:', report['octave'])
with open('b3_probe.json', 'w') as fh:
    json.dump(report, fh, indent=1)
