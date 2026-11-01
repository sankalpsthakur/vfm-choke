# M1 gate: run the physics invariants and keep the report as a kernel output.
import json
import os
import subprocess
import sys
import time

t0 = time.time()
env = dict(os.environ, PYTHONPATH='src')
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests', '-rA',
                    '--junitxml=test_report.xml'], capture_output=True, text=True, env=env)
print(r.stdout[-30000:])
print(r.stderr[-5000:])
lines = r.stdout.strip().splitlines()
summary = {'returncode': r.returncode, 'seconds': round(time.time() - t0, 1), 'summary': lines[-1] if lines else ''}
with open('test_report.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
print(summary)
