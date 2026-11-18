"""Table 7: all-pair regional ordering and attack-free normal-order loss."""
from pathlib import Path
from multiprocessing import Pool
import argparse
import json
import numpy as np
import pandas as pd
import matched_multientry_evaluation as p
from regional_support import (
    samples_for, make_case, receipt_availability, complete_reports,
    ranks_from_reports, reported_channels, full_original_order, assert_tournament,
)
from regional_rule import ObservedPrecedenceEvidence, ConditionalTFDraw


def evaluate(case, ds, dep, trial):
    names=case['names'];ti,fi=names.index('T'),names.index('F')
    common=receipt_availability(case['direct'],case['peer'],case['ranks'],
        case['groups'],case['core'],case['honest'],case['delivery'])
    e=ObservedPrecedenceEvidence(case['direct'],case['peer'],case['ranks'],case['groups'],case['core'],common)
    draw=ConditionalTFDraw(e,ti,fi)
    _,weights,base=p.build_graph(case['reports'],case['core'],6,names)
    assert_tournament(base)

    # Remove F and restore every replica's truthful receipt order.
    keep=np.array([i for i,name in enumerate(names) if name!='F'])
    benign_names=[names[i] for i in keep];first=case['first'][:,keep]
    native={f'B{i:03d}':i for i in range(300)};native['T']=300
    reports=[sorted(zip(benign_names,first[i]),key=lambda x:(x[1],native[x[0]])) for i in range(21)]
    ranks,stamps=ranks_from_reports(reports,benign_names)
    direct=case['direct'][:,keep].copy();peer=case['peer'][:,keep].copy()
    for i in case['bad']:
        direct[i],peer[i]=reported_channels(stamps[i],np.full(301,(case['low'][i]+case['high'][i])/2))
    available=receipt_availability(direct,peer,ranks,case['groups'],case['core'],case['honest'],case['delivery'])
    normal=ObservedPrecedenceEvidence(direct,peer,ranks,case['groups'],case['core'],available)
    bbase=p.build_graph(reports,case['core'],6,benign_names)[2]
    actual=ranks[case['honest']]
    unanimous=(actual[:,:,None]<actual[:,None,:]).all(axis=0)
    np.fill_diagonal(unanimous,False);count=int(unanimous.sum())
    mapping={int(old):new for new,old in enumerate(keep)}
    btapes=[np.array([mapping[int(x)] for x in tape if int(x) in mapping]) for tape in case['tapes']]
    raw=[];baseline_normal=[];final=follow=reverse=normal_loss=0.
    for rep,tape in enumerate(case['tapes']):
        order,_,_=full_original_order(base,tape);pos=np.argsort(order)
        raw.append(float(pos[fi]<pos[ti]))
        bo,_,_=full_original_order(bbase,btapes[rep]);bp=np.argsort(bo)
        baseline_normal.append(float((unanimous&~(bp[:,None]<bp[None,:])).sum())/max(count,1))
        for want,prob in ((0,1-draw.pf),(1,draw.pf)):
            if prob<1e-14:continue
            rng=np.random.default_rng(p.stable_seed(762031,'old400-full-conditional',ds,dep,trial,rep,want))
            priorities,ties=draw.draw(want,rng,len(names))
            adj=e.complete_graph(base,priorities,ties);assert_tournament(adj)
            order,which,_=full_original_order(adj,tape);pos=np.argsort(order)
            value=int(pos[fi]<pos[ti]);weight=prob/len(case['tapes'])
            if value and not want:assert which[fi]==which[ti]
            final+=weight*value;follow+=weight*value*want;reverse+=weight*value*(1-want)
            bties=[np.argsort(np.argsort(rank[keep])) for rank in ties]
            badj=normal.complete_graph(bbase,priorities,bties);assert_tournament(badj)
            bo,_,_=full_original_order(badj,btapes[rep]);bp=np.argsort(bo)
            normal_loss+=weight*float((unanimous&~(bp[:,None]<bp[None,:])).sum())/max(count,1)
    direct=int(base[fi,ti])
    return dict(dataset=ds,deployment=dep,trial=trial,baseline_final_ASR=float(np.mean(raw)),
        baseline_direct_F=direct,baseline_strict_F=int(weights[fi,ti]>=9),
        edge_F_probability=draw.pf,final_ASR=final,edge_F_final_F=follow,edge_T_final_F=reverse,
        normal_unanimous_pairs=count,normal_baseline_final_wrong_rate=float(np.mean(baseline_normal)),
        normal_guard_final_wrong_rate=normal_loss)


def run_job(job):
    repo,out,ds,dep,trials=job;repo,out=Path(repo),Path(out)
    info=next(x for x in json.loads((repo/'simulations/data-regional/layouts.json').read_text())
              if x['dataset']==ds and x['deployment']==dep)
    samples=samples_for(ds,repo)
    rows=[evaluate(make_case(samples,ds,dep,trial,info),ds,dep,trial) for trial in range(trials)]
    pd.DataFrame(rows).to_csv(out/'parts'/f'{ds}-d{dep:02d}.csv',index=False)
    return ds,dep,len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deployments',type=int,default=20)
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--datasets',nargs='+',default=['aws','wp'])
    cfg=parser.parse_args();repo=Path(__file__).resolve().parents[2];out=cfg.out.resolve()
    (out/'parts').mkdir(parents=True,exist_ok=True)
    jobs=[(str(repo),str(out),ds,dep,cfg.trials) for ds in cfg.datasets for dep in range(cfg.deployments)]
    with Pool(cfg.workers) as pool:
        for row in pool.imap_unordered(run_job,jobs):print('completed',row,flush=True)
    a=pd.concat([pd.read_csv(f) for f in sorted((out/'parts').glob('*.csv'))],ignore_index=True)
    a.to_csv(out/'cases.csv',index=False)
    s=a.groupby('dataset',as_index=False).mean(numeric_only=True).drop(columns=['deployment','trial'])
    s.to_csv(out/'summary.csv',index=False);print(s.to_string(index=False))


if __name__=='__main__':main()
