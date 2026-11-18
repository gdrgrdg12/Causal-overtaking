"""Causal p=1 gossip controls on the ORIGINAL attack layouts, without PoPs.

Targets, background traffic and F share one event simulation. A corrupt entry
constructs F only after its replica actually learns T. Dominated duplicate
delivery events are omitted exactly, but sent copies are counted in costs.
"""
from pathlib import Path
from multiprocessing import Pool
import argparse
import heapq
import json
import time

import numpy as np
import pandas as pd

from common import p, Replay, args_for, Samples


class CausalGossip:
    def __init__(self,bg,t,bad,flinks,decisions,wire,batch,adversarial=False,rho=0.):
        self.n=len(t);self.bad=set(map(int,bad));self.bad_order=list(map(int,bad))
        self.bad_pos={b:k for k,b in enumerate(self.bad_order)}
        self.flinks=flinks;self.decisions=decisions;self.wire=wire
        self.batch=batch;self.adversarial=adversarial;self.rho=rho
        self.first=np.full((302,self.n),np.inf);self.best=self.first.copy()
        self.sent=np.zeros(self.n,bool);self.flush=np.zeros((self.n,self.n),int)
        self.queues=[[[] for _ in range(self.n)] for _ in range(self.n)]
        self.events=[];self.serial=0;self.messages=self.copies=0
        self.wait_sum=self.max_wait=0.;self.first_construct=np.inf
        for tx in range(300):
            for node in range(self.n):self.push(bg[tx,node],node,tx)
        for node in range(self.n):self.push(t[node],node,300)

    def push(self,when,node,tx,construct=False):
        when=float(when)
        if not construct:
            if when>=self.best[tx,node]:return
            self.best[tx,node]=when
        self.serial+=1
        heapq.heappush(self.events,(when,int(tx),int(construct),self.serial,int(node)))

    def broadcast_f(self,node,when):
        if self.sent[node]:return
        self.sent[node]=True
        for peer,delay in enumerate(self.flinks[self.bad_pos[node]]):
            self.push(when+delay,peer,301)

    def gossip(self,node,tx,when):
        peers=np.flatnonzero(self.decisions[tx,node])
        if self.batch==1:
            counters=self.flush[node,peers]
            delay=self.wire[counters,node,peers]
            self.flush[node,peers]+=1;self.messages+=len(peers);self.copies+=len(peers)
            arrivals=when+delay
            improving=peers[arrivals<self.best[tx,peers]]
            for peer in improving:
                self.push(when+self.wire[self.flush[node,peer]-1,node,peer],peer,tx)
            return
        for peer in peers:
            queue=self.queues[node][peer];queue.append((tx,when))
            if len(queue)<self.batch:continue
            count=self.flush[node,peer];self.flush[node,peer]+=1
            delay=self.wire[count,node,peer];self.messages+=1;self.copies+=len(queue)
            for offset,(body,seen) in enumerate(queue):
                self.wait_sum+=when-seen;self.max_wait=max(self.max_wait,when-seen)
                self.push(when+delay+offset*.001,peer,body)
            queue.clear()

    def run(self):
        while self.events:
            when,tx,construct,_,node=heapq.heappop(self.events)
            if construct:
                assert node in self.bad and np.isfinite(self.first[300,node])
                assert when>=self.first[300,node]+self.rho-1e-9
                self.first_construct=min(self.first_construct,when)
                self.broadcast_f(node,when)
                continue
            if when>self.best[tx,node] or np.isfinite(self.first[tx,node]):continue
            self.first[tx,node]=when
            if node in self.bad and tx==300:
                self.push(when+self.rho,node,301,construct=True)
            if self.adversarial and node in self.bad:
                if tx==301:self.broadcast_f(node,when)
                continue
            self.gossip(node,tx,when)
        assert np.isfinite(self.first).all()
        assert self.first[301].min()>=self.first_construct-1e-9
        return self.first,dict(gossip_messages=self.messages,gossip_copies=self.copies,
            mean_batch_wait_ms=self.wait_sum/max(self.copies,1),max_batch_wait_ms=self.max_wait,
            unflushed_copies=sum(len(q) for row in self.queues for q in row),
            earliest_F_construction=self.first_construct)


_AWS=None


def gossip_wire(trace,d,ds,dep,trial,repo):
    global _AWS
    sites=[p.physical_location(r) for r in d['replicas']];n=len(sites)
    if ds=='wp':
        ix=[trace.location_index[s] for s in sites]
        matrix=trace.snapshot_matrices[trace.active_snapshot['execution']][np.ix_(ix,ix)]
        return np.broadcast_to(matrix,(302,n,n))
    if _AWS is None:_AWS=Samples(repo/'simulations/data-ping/aws-individual_pings.csv',762031)
    ix=np.array([_AWS.index[s] for s in sites])
    uniform=np.random.default_rng(p.stable_seed(762031,'full-gossip-wire',ds,n,dep,trial)).random((302,n,n))
    indices=(uniform*_AWS.counts[1,ix[:,None],ix[None,:]]).astype(int)
    return _AWS.samples[1,ix[None,:,None],ix[None,None,:],indices]


def replay(first,bad,core,args,ds,dep,trial,validate=False):
    names=[f'B{j:03d}' for j in range(300)]+['T','F']
    reports=[sorted(zip(names,first[:,i]),key=lambda x:x[1]) for i in range(args.num_nodes)]
    for i in bad:reports[i]=p.forge_byzantine_report(reports[i],'pair-swap')
    r=Replay(reports,args,ds,dep,trial);result=r.evaluate(core,validate=validate)
    honest=[i for i in range(args.num_nodes) if i not in set(bad)]
    selected=[i for i in core if i not in set(bad)]
    result.update(honest_Ffirst=int((first[301,honest]<first[300,honest]).sum()),
                  selected_honest_Ffirst=int((first[301,selected]<first[300,selected]).sum()))
    return result


def run_job(job):
    repo,out,ds,n,dep,trials=job;repo,out=Path(repo),Path(out);start=time.monotonic()
    args=args_for(ds,repo);f=(n-1)//4
    args.num_nodes=n;args.corrupted_observers=f;args.byzantine_reports=f
    args.replica_regions=f;args.entry_counts=[f]
    trace=p.LatencyTrace(args.ping_files[0],'half',.15)
    d=p.make_deployment(trace,dep,args);bad=d['corrupted_replica_indices'];core=d['reporter_order'][:n-f]
    rows=[]
    modes=[('plain',0.,1,False),('p01-b5',.1,5,False),('p1-b5',1.,5,False),('p1-b1',1.,1,False),
           ('adversarial-p01-b5',.1,5,True),('adversarial-p1-b5',1.,5,True),('adversarial-p1-b1',1.,1,True)]
    for trial in range(trials):
        snap=p.trial_snapshot(trace,d,trial,args)
        t=np.array([snap['t_arrivals'][r] for r in d['replicas']])
        records=[dict(snap['direct_without_f'][r]) for r in d['replicas']]
        bg=np.array([[record[f'B{j:03d}'] for record in records] for j in range(300)])
        flinks=np.array([[snap['singleton_arrivals'][e['id']][r] for r in d['replicas']]
                        for e in d['entries']])-t[np.array(bad),None]
        assert np.min(flinks)>-1e-8;flinks=np.maximum(flinks,0.)
        uniform=np.random.default_rng(p.stable_seed(args.seed,ds,dep,trial,'gossip-decisions-pcg64-v1')).random((302,n,n))
        wire=gossip_wire(trace,d,ds,dep,trial,repo)
        for mode,prob,batch,adversarial in modes:
            decisions=uniform<=prob
            diagonal=np.arange(n);decisions[:,diagonal,diagonal]=False
            sim=CausalGossip(bg,t,bad,flinks,decisions,wire,batch,adversarial)
            first,cost=sim.run()
            result=replay(first,bad,core,args,ds,dep,trial,validate=trial==0)
            if mode=='plain':
                expected=np.array([min(snap['singleton_arrivals'][e['id']][r] for e in d['entries']) for r in d['replicas']])
                np.testing.assert_allclose(first[:300],bg,rtol=0,atol=1e-9)
                np.testing.assert_allclose(first[300],t,rtol=0,atol=1e-9)
                np.testing.assert_allclose(first[301],expected,rtol=0,atol=1e-9)
            rows.append(dict(dataset=ds,n=n,deployment=dep,trial=trial,mode=mode,gossip_p=prob,batch=batch,
                adversarial_forwarding=adversarial,**result,**cost))
    label=f'{ds}-n{n}-d{dep:02d}'
    pd.DataFrame(rows).to_csv(out/'parts'/f'{label}.csv',index=False)
    return label,len(rows),round(time.monotonic()-start,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deployments',type=int,default=20)
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--sizes',type=int,nargs='+',default=[21])
    parser.add_argument('--datasets',nargs='+',default=['aws','wp'])
    parser.add_argument('--workers',type=int,default=4)
    cfg=parser.parse_args();repo=Path(__file__).resolve().parents[2];out=cfg.out.resolve()
    (out/'parts').mkdir(parents=True,exist_ok=True)
    jobs=[(str(repo),str(out),ds,n,dep,cfg.trials) for ds in cfg.datasets for n in cfg.sizes for dep in range(cfg.deployments)]
    with Pool(cfg.workers) as pool:
        for result in pool.imap_unordered(run_job,jobs):print('completed',result,flush=True)
    a=pd.concat([pd.read_csv(f) for f in sorted((out/'parts').glob('*.csv'))],ignore_index=True)
    a.to_csv(out/'cases.csv',index=False)
    metrics=['strict_edge','final_success','joint_success','honest_Ffirst','selected_honest_Ffirst',
             'gossip_messages','gossip_copies','mean_batch_wait_ms','max_batch_wait_ms']
    s=a.groupby(['dataset','n','mode'],as_index=False)[metrics].mean()
    s.to_csv(out/'summary.csv',index=False)
    print(s[['dataset','n','mode','strict_edge','final_success','mean_batch_wait_ms']].to_string(index=False),flush=True)


if __name__=='__main__':main()
