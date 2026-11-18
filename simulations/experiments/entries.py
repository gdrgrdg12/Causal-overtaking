"""Tables 3 and 9: one/five observers, fixed reporters, nested remote entries."""
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

from common import Replay, args_for, intervals
COUNTS=[1,2,5,10,20]

def components(trace,d,phase,trial,args):
    replicas=d['replicas'];dep=d['deployment'];bad=d['corrupted_replica_indices']
    locations=trace.locations;by_site={p.physical_location(replicas[b]):k for k,b in enumerate(bad)}
    assert len(by_site)==5
    key='train' if phase=='history' else 'trial'
    trace.set_snapshot_context(phase,args.seed,key,dep,trial)
    if phase=='history':
        victim=d['clients'][trial%len(d['clients'])]
        target=np.array([p.T_GEN_TIME+trace.sample(phase,victim,r,args.seed,key,dep,trial,'T',r) for r in replicas])
    else:
        victim=p.stable_rng(args.seed,trace.dataset,dep,trial,'victim').choice(d['clients'])
        target=np.array([p.T_GEN_TIME+trace.sample(phase,victim,r,args.seed,key,dep,trial,'T',r) for r in replicas])
    notify=np.empty((5,len(locations)));delivery=np.empty((len(locations),21))
    for e,site in enumerate(locations):
        # Reuse native source labels at the five original co-located sites:
        # the all-five direct control then reproduces the paper exactly.
        label=f'C{by_site[site]:03d}' if site in by_site else f'X{e:03d}'
        source=replicas[bad[by_site[site]]] if site in by_site else site
        for j,r in enumerate(replicas):
            delivery[e,j]=trace.sample(phase,source,r,args.seed,key,dep,trial,'submit',label,r)
        for k,b in enumerate(bad):
            notify[k,e]=0. if p.physical_location(replicas[b])==site else trace.sample(
                phase,replicas[b],site,args.seed,'entry-scaling-notify',dep,trial,k,e)
    arrivals=target[np.array(bad),None,None]+args.rho_ms+notify[:,:,None]+delivery[None,:,:]
    assert np.all(arrivals>=target[np.array(bad),None,None]-1e-9)
    return dict(target=target,notify=notify,delivery=delivery,paths=arrivals,victim=victim)


def train_orders(trace,d,args):
    history=[components(trace,d,'history',j,args) for j in range(24)]
    bad=d['corrupted_replica_indices'];core=d['reporter_order'][:16]
    selected=np.array([i for i in core if i not in bad])
    mean_observation=np.mean([h['target'][bad] for h in history],axis=0)
    observer_order=np.argsort(mean_observation,kind='stable')
    orders={}
    for family,observers in [('one-observer',observer_order[:1])]:
        winner=np.stack([np.min(h['paths'][observers],axis=0)[:,selected]<h['target'][selected][None,:] for h in history])
        # [candidate entry, training workload x selected honest receiver]
        sets=winner.transpose(1,0,2).reshape(len(trace.locations),-1)
        uncovered=np.ones(sets.shape[1],bool);remaining=set(range(len(trace.locations)));order=[]
        # Geographical spread is a secondary historical-latency criterion,
        # never a fabricated coordinate or current-test optimization.
        while remaining and len(order)<20:
            chosen=max(remaining,key=lambda e:(int(np.sum(sets[e]&uncovered)),int(sets[e].sum()),
                min((trace.historical_distance(trace.locations[e],trace.locations[k]) for k in order),default=0.),-e))
            order.append(chosen);uncovered&=~sets[chosen];remaining.remove(chosen)
        orders[family]=order
    # Match entry locations as well as K across the one/five-observer arms.
    # Otherwise separately optimized entry lists confound the observer effect.
    orders['five-observers']=orders['one-observer'].copy()
    return observer_order,orders,mean_observation


def evaluate_fixed_logs(snapshot,f,bad,frozen,core,args,ds,dep,trial,cache,validate=False):
    replicas=list(snapshot['direct_without_f'])
    # The snapshot dict is constructed in replica order in the original code.
    base=[snapshot['direct_without_f'][r] for r in replicas]
    times=[np.array(sorted(v for _,v in row)) for row in base]
    hcore=[i for i in core if i not in bad]
    cache_key=tuple(int(np.searchsorted(times[i],f[i],side='right')) for i in hcore)
    if cache_key in cache and not validate:return cache[cache_key]
    reports=[sorted(row+[('F',float(f[i]))],key=lambda x:x[1]) for i,row in enumerate(base)]
    for i in bad:reports[i]=list(frozen[i])
    assert all(reports[i]==frozen[i] for i in bad)
    result=Replay(reports,args,ds,dep,trial).evaluate(core,validate=validate)
    h=np.array([i for i in range(21) if i not in bad]);t=np.array([snapshot['t_arrivals'][r] for r in replicas])
    result.update(honest_Ffirst=int(np.sum(f[h]<t[h])),selected_honest_Ffirst=int(np.sum(f[hcore]<t[hcore])))
    assert result['F_votes']==5+result['selected_honest_Ffirst']
    assert result['strict_edge']==int(result['selected_honest_Ffirst']>=4)
    # Cache graph metrics only: unselected honest support is recomputed below.
    cache[cache_key]={k:v for k,v in result.items() if k!='honest_Ffirst'}
    return cache[cache_key]


def run_dataset(job):
    ds,cfg=job;repo=Path(cfg['repo']);out=Path(cfg['out']);started=time.monotonic()
    args=args_for(ds,repo);trace=p.LatencyTrace(args.ping_files[0],'half',.15)
    d=p.make_deployment(trace,cfg['deployment'],args);dep=d['deployment'];bad=d['corrupted_replica_indices']
    core=d['reporter_order'][:16];honest=np.array([i for i in range(21) if i not in bad])
    hcore=np.array([i for i in core if i not in bad]);o_order,orders,means=train_orders(trace,d,args)
    locidx={s:i for i,s in enumerate(trace.locations)}
    native_entries=[locidx[p.physical_location(d['replicas'][bad[k]])] for k in range(5)]
    metadata=dict(dataset=ds,deployment=dep,replicas=d['replicas'],corrupted=bad,reporters=core,
        observer_order_positions=o_order.tolist(),observer_order_ids=[bad[k] for k in o_order],
        historical_mean_observer_arrival_ms=means.tolist(),entry_orders={k:[trace.locations[i] for i in order] for k,order in orders.items()},
        original_direct_entry_positions=native_entries,locations=trace.locations)
    (out/f'{ds}-deployment.json').write_text(json.dumps(metadata,indent=2))
    rows=[];single_rows=[];report_hashes=[];pool_rows=[];validation_checks=0
    for trial in range(cfg['trials']):
        snap=p.trial_snapshot(trace,d,trial,args);x=components(trace,d,'execution',trial,args)
        t=np.array([snap['t_arrivals'][r] for r in d['replicas']]);np.testing.assert_array_equal(t,x['target'])
        assert list(snap['direct_without_f'])==list(d['replicas'])
        native=np.array([[snap['singleton_arrivals'][e['id']][r] for r in d['replicas']] for e in d['entries']])
        for k in range(5):np.testing.assert_allclose(x['paths'][k,native_entries[k]],native[k],rtol=0,atol=1e-9)
        base_f=native.min(axis=0)
        reference=[sorted(snap['direct_without_f'][r]+[('F',float(base_f[i]))],key=lambda v:v[1]) for i,r in enumerate(d['replicas'])]
        frozen={i:p.forge_byzantine_report(reference[i],'pair-swap') for i in bad}
        cache={}
        def evaluate(f,validate=False):
            nonlocal validation_checks
            r=evaluate_fixed_logs(snap,f,bad,frozen,core,args,ds,dep,trial,cache,validate)
            validation_checks+=int(validate)
            return r|dict(honest_Ffirst=int(np.sum(f[honest]<t[honest])))
        baseline=evaluate(base_f,trial==0)
        common=dict(dataset=ds,deployment=dep,trial=trial,victim=x['victim'])
        for family,observers in [('one-observer',o_order[:1]),('five-observers',o_order)]:
            arrival=np.min(x['paths'][observers],axis=0);order=orders[family]
            release=np.min(t[np.array(bad)[observers],None]+x['notify'][observers],axis=0)
            winning_observer=np.argmin(t[np.array(bad)[observers],None]+x['notify'][observers],axis=0)
            winning_hop=x['notify'][observers[winning_observer],np.arange(len(trace.locations))]
            # Diagnostic oracle over the FULL pool needs no graph replay for
            # support/strictness; final-order oracle is only over each K set.
            all_support=(arrival[:,honest]<t[honest]).sum(axis=1)
            selected_support=(arrival[:,hcore]<t[hcore]).sum(axis=1)
            pool_rows.append(common|dict(family=family,candidates=len(trace.locations),
                best_pool_single_honest=int(all_support.max()),best_pool_single_selected=int(selected_support.max()),
                any_pool_single_strict=int(selected_support.max()>=4)))
            singles=[]
            for e in order:
                r=evaluate(arrival[e]);singles.append(r)
                single_rows.append(common|dict(family=family,entry=trace.locations[e],notification_delay_ms=float(winning_hop[e]),
                    entry_activation_lag_ms=float(release[e]-np.min(t[np.array(bad)[observers]])),**r))
            previous=np.full(21,np.inf)
            for count in COUNTS:
                selected=order[:count];f=np.min(arrival[selected],axis=0)
                assert np.all(f<=previous+1e-9);previous=f
                result=evaluate(f,trial==0 and count in (1,20))
                best_h=max(s['honest_Ffirst'] for s in singles[:count])
                best_sel=max(s['selected_honest_Ffirst'] for s in singles[:count])
                best_strict=max(s['strict_edge'] for s in singles[:count])
                best_final=max(s['final_success'] for s in singles[:count])
                rows.append(common|dict(family=family,observers=len(observers),K=count,
                    entries=';'.join(trace.locations[e] for e in selected),
                    best_single_honest=best_h,best_single_selected=best_sel,best_single_strict=best_strict,best_single_final=best_final,
                    extra_honest=result['honest_Ffirst']-best_h,extra_selected=result['selected_honest_Ffirst']-best_sel,
                    strict_composition_only=int(result['strict_edge'] and not best_strict),
                    final_gain_over_best_single=result['final_success']-best_final,
                    mean_notification_ms=float(np.mean(winning_hop[selected])),
                    mean_activation_lag_ms=float(np.mean(release[selected]-np.min(t[np.array(bad)[observers]]))),**result))
        # The original one-co-located-entry-per-observer counterfactual.
        individual=[evaluate(native[k]) for k in o_order]
        previous=np.full(21,np.inf)
        for count in (1,2,5):
            active=o_order[:count];f=native[active].min(axis=0)
            assert np.all(f<=previous+1e-9);previous=f
            result=evaluate(f,trial==0)
            best_h=max(s['honest_Ffirst'] for s in individual[:count]);best_sel=max(s['selected_honest_Ffirst'] for s in individual[:count])
            best_strict=max(s['strict_edge'] for s in individual[:count]);best_final=max(s['final_success'] for s in individual[:count])
            rows.append(common|dict(family='one-entry-per-observer',observers=count,K=count,
                entries=';'.join(p.physical_location(d['replicas'][bad[k]]) for k in active),
                best_single_honest=best_h,best_single_selected=best_sel,best_single_strict=best_strict,best_single_final=best_final,
                extra_honest=result['honest_Ffirst']-best_h,extra_selected=result['selected_honest_Ffirst']-best_sel,
                strict_composition_only=int(result['strict_edge'] and not best_strict),
                final_gain_over_best_single=result['final_success']-best_final,mean_notification_ms=0.,mean_activation_lag_ms=0.,**result))
        if trial%5==4:print(ds,'trials',trial+1,'seconds',round(time.monotonic()-started,1),flush=True)
    pd.DataFrame(rows).to_csv(out/f'{ds}-cases.csv',index=False)
    pd.DataFrame(single_rows).to_csv(out/f'{ds}-singletons.csv',index=False)
    pd.DataFrame(pool_rows).to_csv(out/f'{ds}-full-pool-singleton-support.csv',index=False)
    return ds,len(rows),validation_checks,round(time.monotonic()-started,2)


def run_job(job):
    repo,out,ds,dep,trials=job
    part=Path(out)/'parts'/f'd{dep:02d}';part.mkdir(parents=True,exist_ok=True)
    return run_dataset((ds,dict(repo=repo,out=str(part),deployment=dep,trials=trials)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deployments',type=int,default=20)
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--datasets',nargs='+',default=['aws','wp'])
    parser.add_argument('--bootstrap-repeats',type=int,default=10000)
    cfg=parser.parse_args();repo=Path(__file__).resolve().parents[2];out=cfg.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    jobs=[(str(repo),str(out),ds,dep,cfg.trials) for ds in cfg.datasets for dep in range(cfg.deployments)]
    with Pool(cfg.workers) as pool:
        for row in pool.imap_unordered(run_job,jobs):print('completed',row,flush=True)
    from summarize import summarize_entries
    summarize_entries(out,cfg.bootstrap_repeats)


if __name__=='__main__':main()
