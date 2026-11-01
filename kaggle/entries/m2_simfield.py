# M2: simulate the SimField population and keep its tables as kernel outputs for later milestones.
import json
import pickle
import sys
import time

sys.path.insert(0, 'src')
from vfmchoke import simfield  # noqa: E402

t0 = time.time()
pop = simfield.sample_population(n_groups=200, max_wells=10, seed=2026)
print('population:', int(pop.mask.sum()), 'wells in', pop.mask.shape[0], 'groups', flush=True)
res = simfield.simulate(pop, days=180, seed=2026, log_every=10)
with open('simfield_pop.pkl', 'wb') as fh:
    pickle.dump(pop, fh)
simfield.wells_frame(pop).to_parquet('simfield_wells.parquet', index=False)
res['daily'].to_parquet('simfield_daily.parquet', index=False)
res['tests'].to_parquet('simfield_tests.parquet', index=False)
res['groups'].to_parquet('simfield_groups.parquet', index=False)
summary = simfield.summarize(res)
summary['seconds'] = round(time.time() - t0, 1)
with open('m2_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
