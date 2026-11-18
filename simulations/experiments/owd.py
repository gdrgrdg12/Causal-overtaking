"""Original n21 attack on measured ESnet OWAMP distributions.

History and execution come from disjoint days. No Guard, new relay, admission
delay, extra API, or finalizer changes. Client positions are taken either from
the SAME measured address pool (original behavior) or actual replica sites.
"""
from pathlib import Path
from multiprocessing import Pool
import argparse
import json
import time

import numpy as np
import pandas as pd

from common import Replay, args_for, p


SCOUT = 'simulations/data-owd'


class OWAMPTrace:
    def __init__(self, folder, view='measured-owd'):
        self.dataset = 'esnet'
        self.view = view
        self.histograms = {}
        self.means = {}
        self.times = {}
        folder = Path(folder)
        sources = {'history': 'esnet-29node-history.json',
                   'execution': 'esnet-29node-snapshot.json'}
        for phase, filename in sources.items():
            raw = json.loads((folder / filename).read_text())
            assert not raw.get('timed_out', False)
            assert raw.get('_shards', {}).get('failed', 0) == 0
            mapping, means, times = {}, {}, {}
            buckets = raw['aggregations']['sources']
            assert buckets.get('sum_other_doc_count', 0) == 0
            for source in buckets['buckets']:
                assert source['destinations'].get('sum_other_doc_count', 0) == 0
                for dest in source['destinations']['buckets']:
                    item = dest['latest']['hits']['hits'][0]['_source']
                    assert item['pscheduler']['tool'] == 'powstream'
                    assert item['meta']['ip_version'] == 4
                    assert item['result']['succeeded']
                    pair = (item['test']['spec']['source'], item['test']['spec']['dest'])
                    hist = item['result']['latency']['histogram']
                    values = np.asarray(hist['values'], float)
                    counts = np.asarray(hist['counts'], np.int64)
                    assert np.all(values > 0) and np.all(counts > 0)
                    order = np.argsort(values)
                    values, counts = values[order], counts[order]
                    cumulative = counts.cumsum()
                    mapping[pair] = (values, cumulative, int(cumulative[-1]))
                    means[pair] = float(np.dot(values, counts) / counts.sum())
                    times[pair] = item['pscheduler']['start_time']
            self.histograms[phase], self.means[phase], self.times[phase] = mapping, means, times
        self.locations = sorted({u for pair in self.histograms['history'] for u in pair})
        for phase in sources:
            assert len(self.histograms[phase]) == len(self.locations) * (len(self.locations) - 1)
            assert all((s, d) in self.histograms[phase]
                       for s in self.locations for d in self.locations if s != d)
        assert max(self.times['history'].values()) < min(self.times['execution'].values())

    @property
    def is_hourly_snapshot(self):
        return False

    def set_snapshot_context(self, phase, seed, *key):
        pass

    def active_snapshot_hour(self, phase):
        return '2026-11-16T04:50/05:00Z' if phase == 'history' else '2026-11-17T04:50/05:00Z'

    def _draw(self, phase, source, destination, seed, key):
        if source == destination:
            # Local delivery, not an imputed Internet OWD or extra relay.
            return 0.
        values, cumulative, total = self.histograms[phase][source, destination]
        rng = p.stable_rng(seed, 'latency-index', phase, source, destination, *key)
        return float(values[np.searchsorted(cumulative, rng.randrange(total), side='right')])

    def sample(self, phase, source, destination, seed, *key):
        source, destination = p.physical_location(source), p.physical_location(destination)
        value = self._draw(phase, source, destination, seed, key)
        if self.view == 'symmetric-from-owd':
            value = .5 * (value + self._draw(phase, destination, source, seed, key))
        elif self.view != 'measured-owd':
            raise ValueError(self.view)
        return value

    def historical_distance(self, source, destination):
        source, destination = p.physical_location(source), p.physical_location(destination)
        if source == destination:
            return 0.
        return .5 * (self.means['history'][source, destination]
                     + self.means['history'][destination, source])


def make_args(repo, layout):
    args = args_for('aws', repo)
    args.replica_layout = layout
    return args


def run_job(job):
    repo, out, layout, view, dep_id, trials = job
    repo, out = Path(repo), Path(out)
    start = time.monotonic()
    trace = OWAMPTrace(repo / SCOUT, view)
    args = make_args(repo, layout)
    deployment = p.make_deployment(trace, dep_id, args)
    original_clients = list(deployment['clients'])
    actual_sites = sorted({p.physical_location(r) for r in deployment['replicas']})
    # Location choice only: no knowledge of which identities are honest.
    local_clients = p.stable_rng(args.seed, trace.dataset, 'replica-site-clients', dep_id).sample(
        actual_sites, args.num_clients)
    policies = ['same-address-pool', 'replica-addresses-only'] if view == 'measured-owd' else ['same-address-pool']
    rows = []
    validated = 0
    for policy in policies:
        deployment['clients'] = original_clients if policy == 'same-address-pool' else local_clients
        assert policy != 'replica-addresses-only' or set(deployment['clients']) <= set(actual_sites)
        for trial in range(trials):
            snap = p.trial_snapshot(trace, deployment, trial, args)
            f_arrival = p.union_arrivals(snap['singleton_arrivals'], [e['id'] for e in deployment['entries']])
            physical = p.apply_dissemination(
                trace, deployment, trial, snap['direct_without_f'], f_arrival, 'plain', args)
            reports, byzantine = p.apply_byzantine_reports(
                [physical[r] for r in deployment['replicas']], deployment, args)
            bad = set(deployment['corrupted_replica_indices'])
            assert len(bad) == len(byzantine) == 5
            assert all(p.precedes(reports[i], 'F', 'T') for i in bad)
            replay = Replay(reports, args, trace.dataset, dep_id, trial)
            reporters = deployment['reporter_order'][:16]
            check = trial == 0
            result = replay.evaluate(reporters, validate=check)
            validated += int(check)
            direct = int(result['F_votes'] > result['T_votes'] or (
                result['F_votes'] == result['T_votes'] and replay.fi < replay.ti))
            selected_honest = [i for i in reporters if i not in bad]
            selected_front = sum(f_arrival[deployment['replicas'][i]] < snap['t_arrivals'][deployment['replicas'][i]]
                                 for i in selected_honest)
            all_front = sum(f_arrival[r] < snap['t_arrivals'][r]
                            for i, r in enumerate(deployment['replicas']) if i not in bad)
            assert result['F_votes'] == 5 + selected_front
            rows.append(dict(dataset='esnet', layout=layout, view=view, clients=policy,
                             deployment=dep_id, trial=trial, victim=snap['victim'],
                             victim_at_replica=int(snap['victim'] in actual_sites),
                             accepted_bad=len(set(reporters) & bad),
                             direct_F_edge=direct, selected_honest_Ffirst=selected_front,
                             all_honest_Ffirst=all_front,
                             direct_F_final_F=direct * result['final_success'],
                             direct_T_final_F=(1-direct) * result['final_success'], **result))
    label = f'{layout}-{view}-d{dep_id:02d}'
    pd.DataFrame(rows).to_csv(out / 'parts' / f'{label}.csv', index=False)
    metadata = dict(layout=layout, view=view, deployment=dep_id,
                    replicas=deployment['replicas'], original_clients=original_clients,
                    replica_address_clients=local_clients, physical_replica_sites=actual_sites,
                    bad=sorted(bad), reporters=reporters,
                    training=deployment['corrupted_placement_info'], validation_replays=validated,
                    seconds=time.monotonic()-start)
    (out / 'parts' / f'{label}.json').write_text(json.dumps(metadata, indent=2))
    return label, len(rows), round(metadata['seconds'], 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deployments', type=int, default=20)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--layouts', nargs='+', default=['global-random', 'latency-clustered'])
    parser.add_argument('--views', nargs='+', default=['measured-owd', 'symmetric-from-owd'])
    cfg = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = cfg.out.resolve()
    (out / 'parts').mkdir(parents=True, exist_ok=True)
    jobs = [(str(repo), str(out), layout, view, dep, cfg.trials)
            for layout in cfg.layouts for view in cfg.views for dep in range(cfg.deployments)]
    with Pool(cfg.workers) as pool:
        for result in pool.imap_unordered(run_job, jobs):
            print('completed', result, flush=True)
    data = pd.concat([pd.read_csv(f) for f in sorted((out / 'parts').glob('*.csv'))], ignore_index=True)
    data.to_csv(out / 'cases.csv', index=False)
    metrics = ['strict_edge', 'direct_F_edge', 'final_success', 'joint_success',
               'selected_honest_Ffirst', 'all_honest_Ffirst', 'direct_F_final_F',
               'direct_T_final_F', 'victim_at_replica']
    summary = data.groupby(['layout', 'view', 'clients'], as_index=False)[metrics].mean()
    counts = data.groupby(['layout', 'view', 'clients'], as_index=False).size().rename(columns={'size': 'workloads'})
    summary = summary.merge(counts)
    summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
