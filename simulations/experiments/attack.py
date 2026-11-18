"""Figure 2 and Tables 4--5: baseline, reporter selection, directional delay."""
from pathlib import Path
from multiprocessing import Pool
from itertools import permutations, product
from math import comb
import argparse
import json
import time
import numpy as np
import pandas as pd
import matched_multientry_evaluation as p

from common import args_for, DirectedTrace, Replay, report_collection, retrain
SEED=762031

def task(job):
    ds,n,dep_id,stage,delta,placement,network_seed,cfg=job
    repo=Path(cfg['repo']);out=Path(cfg['out']);args=args_for(ds,repo)
    f=(n-1)//4;args.num_nodes=n;args.corrupted_observers=f;args.byzantine_reports=f
    args.replica_regions=f;args.entry_counts=[f];args.public_repetitions=20
    base=p.LatencyTrace(args.ping_files[0],'half',.15)
    dep=p.make_deployment(base,dep_id,args)
    trace=base if delta==0 else DirectedTrace(args.ping_files[0],delta,network_seed)
    if delta and placement=='retrained':dep=retrain(trace,dep,args)
    bad=np.asarray(dep['corrupted_replica_indices']);honest=np.array([i for i in range(n) if i not in bad])
    rows=[];checks=0;start=time.monotonic()
    for trial in range(cfg['trials']):
        snap=p.trial_snapshot(trace,dep,trial,args)
        first_f=p.union_arrivals(snap['singleton_arrivals'],[e['id'] for e in dep['entries']])
        real_by_name=p.apply_dissemination(trace,dep,trial,snap['direct_without_f'],first_f,'plain',args)
        real=[real_by_name[r] for r in dep['replicas']]
        reports,_=p.apply_byzantine_reports(real,dep,args)
        replay=Replay(reports,args,ds,dep_id,trial)
        common=dict(dataset=ds,n=n,deployment=dep_id,trial=trial,stage=stage,delta=delta,placement=placement,network_seed=network_seed)
        def add(policy,rr,draw=0,**extra):
            nonlocal checks
            verify=trial==0 and draw==0
            metrics=replay.evaluate(rr,validate=verify);checks+=int(verify)
            rows.append(common|dict(policy=policy,draw=draw,accepted_bad=int(np.isin(rr,bad).sum()),reporters=json.dumps(list(map(int,rr))))|metrics|extra)
        original=np.array(dep['reporter_order'][:n-f]);add('paper-fixed',original)
        if stage=='reporters':
            # Controlled counts: all observers stay active; selection is hidden
            # from the fixed pair-swap program, not chosen from honest orders.
            for b in range(6):
                draws=1 if b==0 else 3
                for draw in range(draws):
                    rng=np.random.default_rng(p.stable_seed(SEED,'supplement-b',ds,dep_id,trial,b,draw))
                    rr=np.r_[rng.choice(bad,b,False),rng.choice(honest,16-b,False)]
                    add('conditional-b',rr,draw,condition_b=b)
            # Uniform benchmark and arrival-generated sets are distinct arms.
            for draw in range(3):
                rng=np.random.default_rng(p.stable_seed(SEED,'supplement-uniform',ds,dep_id,trial,draw))
                add('uniform16',rng.choice(n,16,False),draw)
            leader=int(honest[p.stable_seed(SEED,'supplement-honest-leader',ds,dep_id,trial)%len(honest)])
            transport=np.array([trace.sample('execution',r,dep['replicas'][leader],SEED,'supplement-report-transport',dep_id,trial,i) for i,r in enumerate(dep['replicas'])])
            transport[leader]=0.
            ready=np.array([max(time for _,time in report) for report in real])
            for policy in ('leader-deadline','leader-ready','leader-early-byzantine'):
                rr,send,arrived=report_collection(ready,bad,transport,leader,policy)
                add(policy,rr,leader=leader,accepted_last_ms=float(arrived[rr].max()),
                    local_ready_max_ms=float(ready.max()),send_times=json.dumps(send.tolist()),
                    arrival_times=json.dumps(arrived.tolist()))
    label=f'{stage}-{ds}-n{n}-d{dep_id:02d}-a{delta:g}-{placement}-s{network_seed}'
    pd.DataFrame(rows).to_csv(out/'parts'/f'{label}.csv',index=False)
    info=dict(dataset=ds,n=n,deployment=dep_id,stage=stage,delta=delta,placement=placement,network_seed=network_seed,
        replicas=dep['replicas'],clients=dep['clients'],bad=bad.tolist(),original_reporters=dep['reporter_order'][:n-f],
        placement_training=dep.get('supplement_placement_training',dep.get('placement_info',{})),checks=checks,
        seconds=time.monotonic()-start)
    (out/'parts'/f'{label}.json').write_text(json.dumps(info,indent=2))
    return label,len(rows),checks,round(info['seconds'],2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--deployments',type=int,default=20);parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--stages',nargs='+',default=['baseline','reporters','asymmetry'])
    parser.add_argument('--datasets',nargs='+',default=['aws','wp'])
    parser.add_argument('--sizes',nargs='+',type=int,default=[9,21,101])
    parser.add_argument('--bootstrap-repeats',type=int,default=10000)
    a=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'parts').mkdir(exist_ok=True)
    cfg=dict(repo=str(repo),out=str(a.out.resolve()),trials=a.trials);jobs=[]
    for ds in a.datasets:
        for dep in range(a.deployments):
            if 'baseline' in a.stages:
                for n in a.sizes:jobs.append((ds,n,dep,'baseline',0.,'original',0,cfg))
            if 'reporters' in a.stages:jobs.append((ds,21,dep,'reporters',0.,'original',0,cfg))
            if 'asymmetry' in a.stages:
                for delta in (.1,.25):
                    for placement in ('frozen','retrained'):
                        jobs.append((ds,21,dep,'asymmetry',delta,placement,0,cfg))
    pending=[]
    for job in jobs:
        ds,n,dep,stage,delta,placement,seed,_=job
        label=f'{stage}-{ds}-n{n}-d{dep:02d}-a{delta:g}-{placement}-s{seed}'
        pending.append(job)
    print('jobs',len(jobs),'pending',len(pending),flush=True)
    with Pool(a.workers) as pool:
        for result in pool.imap_unordered(task,pending):print('done',result,flush=True)
    from summarize import summarize_attack
    summarize_attack(a.out,a.bootstrap_repeats)

if __name__=='__main__':main()
