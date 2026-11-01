# M2b: a variant population (wider, shorter, shallower lines; lower separator pressure; smaller chokes) to check
# that M3 holds when more chokes run critical. Same code; only the network ranges differ and keep its tables as kernel outputs for later milestones.
import json
import pickle
import sys
import time

sys.path.insert(0, 'src')
from vfmchoke import simfield  # noqa: E402

t0 = time.time()
pop = simfield.sample_population(n_groups=120, max_wells=10, seed=2027, choke_frac=0.45,
                                 line_d=(0.254, 0.3048, 0.3556), depth=(30, 250), p_sep=(5e5, 15e5),
                                 length=(1000, 8000), flow_margin=50e5)
print('population:', int(pop.mask.sum()), 'wells in', pop.mask.shape[0], 'groups', flush=True)
res = simfield.simulate(pop, days=180, seed=2027, log_every=10)
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
