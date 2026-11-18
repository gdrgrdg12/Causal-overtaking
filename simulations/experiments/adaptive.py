"""Table 8: train and validate 54 causal policies, freeze before test."""
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

from itertools import product
import regional_support as legacy
from regional_support import *
from regional_rule import ObservedPrecedenceEvidence, outcome

def own_gaps(c,policy):
    bad=c['bad'];middle=(c['low'][bad]+c['high'][bad])/2
    if not policy['frame']:return middle,middle
    a,b=c['low'][bad]-.001,c['high'][bad]+.001
    return np.where(np.abs(a)<np.abs(b),a,b),middle


def pair_score(c,policy):
    t,pt,df,f,pf,rt,rf=legacy.physics(c,policy);bad=c['bad']
    dt=c['raw_t_direct'].copy();pt=pt.copy();df=df.copy();pf=pf.copy()
    gt,gf=own_gaps(c,policy)
    dt[bad],pt[bad]=reported_channels(rt[bad],gt)
    df[bad],pf[bad]=reported_channels(rf[bad],gf)
    channels,orders=gather_records(c['groups'],c['honest'],c['core'],dt,pt,df,pf,c['delivery'])
    _,regional=witness_verdict(dt,pt,df,pf,c['groups'],channels,np.zeros(21),'two-copy-disjoint')
    fixed=['F' if v=='T-first' else 'T' if v=='F-first' else None for v in regional]
    return float(report_probability(rt,rf,c['groups'],orders,fixed,3))


def attack_case(c,policy):
    new=legacy.rebuild(c,policy)
    # Same54candidate report fields as before; frame is now only an own-T
    # channel-gap perturbation, not an anomalous-transaction certificate.
    if policy['frame']:
        ti=new['names'].index('T');bad=new['bad'];gap,_=own_gaps(c,policy)
        stamps=np.array([dict(new['reports'][i])['T'] for i in bad])
        new['direct'][bad,ti],new['peer'][bad,ti]=reported_channels(stamps,gap)
    return new


def full_score(c,policy,ds,dep,trial,cache,validate=False):
    pk=legacy.key(policy)
    if ('policy',pk) in cache and not validate:return cache['policy',pk]
    new=attack_case(c,policy);ti,fi=new['names'].index('T'),new['names'].index('F')
    _,weights,base=p.build_graph(new['reports'],new['core'],6,new['names']);assert_tournament(base)
    common=receipt_availability(new['direct'],new['peer'],new['ranks'],new['groups'],new['core'],new['honest'],new['delivery'])
    e=ObservedPrecedenceEvidence(new['direct'],new['peer'],new['ranks'],new['groups'],new['core'],common)
    ek=(e.order_pool.tobytes(),e.ranks.tobytes(),e.region.tobytes(),base.tobytes())
    if ('evidence',ek) in cache and not validate:
        value=cache['evidence',ek].copy()
    else:
        value=outcome(e,base,new['tapes'],ds,dep,trial,ti,fi,validate)
    cache['evidence',ek]=value.copy()
    pair=pair_score(c,policy);assert abs(pair-value['direct_F'])<1e-10
    raw=float(np.mean([ft_final(base,tape,ti,fi,validate and j==0) for j,tape in enumerate(new['tapes'])]))
    result=dict(guard_final=value['final_attack'],guard_direct_F=value['direct_F'],
        edge_T_final_F=value['edge_T_final_F'],edge_F_final_F=value['edge_F_final_F'],
        baseline_final=raw,baseline_strict=int(weights[fi,ti]>=9),
        range_false_T=0,range_detect_F=0)
    cache['policy',pk]=result
    return result


def run_job(job):
    repo,out,ds,dep,trials,nt,nv=job;repo,out=Path(repo),Path(out);start=time.monotonic()
    part=out/'parts'/f'{ds}-d{dep:02d}.csv';meta=part.with_suffix('.json')
    info=next(x for x in json.loads((repo/'simulations/data-regional/layouts.json').read_text())
              if x['dataset']==ds and x['deployment']==dep)
    s=samples_for(ds,repo)
    training=[legacy.inputs(legacy.HistorySamples(s,ds,0),ds,dep,20000+j,info) for j in range(nt)]
    validation=[legacy.inputs(legacy.HistorySamples(s,ds,1),ds,dep,30000+j,info) for j in range(nv)]
    candidates=[dict(relay=relay,wait=wait,late=late,frame=frame) for relay,wait,late,frame in
                product(['withhold','local-fast','exchange-fast'],[0.,1.,5.],[.001,20.,200.],[False,True])]
    scores=[policy|dict(pair_F=float(np.mean([pair_score(c,policy) for c in training]))) for policy in candidates]
    shortlist={legacy.key(legacy.BASE):legacy.BASE}
    for late in (.001,20.,200.):
        ordered=sorted((r for r in scores if r['late']==late),
            key=lambda r:(r['pair_F'],-r['wait'],-['withhold','local-fast','exchange-fast'].index(r['relay'])),reverse=True)
        best={k:ordered[0][k] for k in legacy.BASE};shortlist[legacy.key(best)]=best
    for frame in (False,True):
        probe=dict(relay='exchange-fast',wait=0.,late=20.,frame=frame);shortlist[legacy.key(probe)]=probe
    validation_cache=[{} for _ in validation];validation_rows=[]
    for policy in shortlist.values():
        results=[full_score(c,policy,ds,dep,30000+j,validation_cache[j]) for j,c in enumerate(validation)]
        validation_rows.append(policy|dict(guard_final=float(np.mean([r['guard_final'] for r in results])),
            guard_direct_F=float(np.mean([r['guard_direct_F'] for r in results]))))
    chosen=sorted(validation_rows,key=lambda r:(r['guard_final'],r['guard_direct_F'],-r['wait'],-r['late']),reverse=True)[0]
    chosen={k:chosen[k] for k in legacy.BASE}
    pd.DataFrame(scores).to_csv(out/'training'/f'{ds}-d{dep:02d}-train.csv',index=False)
    pd.DataFrame(validation_rows).to_csv(out/'training'/f'{ds}-d{dep:02d}-validation.csv',index=False)
    selection=dict(selected=chosen,candidate_count=len(candidates),
        validation_candidates=len(shortlist),training_cases=nt,validation_cases=nv,offline_seconds=time.monotonic()-start)
    (out/'training'/f'{ds}-d{dep:02d}-selection.json').write_text(json.dumps(selection,indent=2))
    policies={'fixed-original':legacy.BASE,'local-fast-actual':dict(legacy.BASE,relay='local-fast'),
        'exchange-fast-actual':dict(legacy.BASE,relay='exchange-fast'),
        'history-adaptive':chosen}
    rows=[]
    for trial in range(trials):
        c=legacy.inputs(s,ds,dep,trial,info);cache={}
        for label,policy in policies.items():
            result=full_score(c,policy,ds,dep,trial,cache,validate=trial==0)
            rows.append(dict(dataset=ds,deployment=dep,trial=trial,policy=label,parameters=legacy.key(policy),**result))
        if trial%10==9:print(ds,dep,'workloads',trial+1,'seconds',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(rows).to_csv(part,index=False)
    meta.write_text(json.dumps(dict(selection=selection,elapsed_seconds=time.monotonic()-start),indent=2))
    return ds,dep,len(rows),round(time.monotonic()-start,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deployments',type=int,default=20)
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--datasets',nargs='+',default=['aws','wp'])
    parser.add_argument('--train-cases',type=int,default=8)
    parser.add_argument('--validation-cases',type=int,default=3)
    parser.add_argument('--bootstrap-repeats',type=int,default=10000)
    cfg=parser.parse_args();repo=Path(__file__).resolve().parents[2];out=cfg.out.resolve()
    for folder in ['parts','training']:(out/folder).mkdir(parents=True,exist_ok=True)
    jobs=[(str(repo),str(out),ds,dep,cfg.trials,cfg.train_cases,cfg.validation_cases)
          for ds in cfg.datasets for dep in range(cfg.deployments)]
    with Pool(cfg.workers) as pool:
        for row in pool.imap_unordered(run_job,jobs):print('completed',row,flush=True)
    from summarize import summarize_adaptive
    summarize_adaptive(out,cfg.bootstrap_repeats)


if __name__=='__main__':main()
