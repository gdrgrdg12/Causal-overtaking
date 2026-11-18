"""Shared trace configuration, original graph replay, and bootstrap."""
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

SEED=762031

def args_for(dataset, repo):
    path = repo / 'simulations/data-ping' / ('aws-individual_pings.csv' if dataset == 'aws' else 'wp-hourly-2020-07-19-20.npz')
    return p.build_parser().parse_args([
        '--ping_files', str(path), '--num_nodes', '21', '--num_clients', '15',
        '--replica_layout', 'global-random' if dataset == 'aws' else 'latency-clustered',
        '--replica_regions', '5', '--observation_model', 'corrupted-replica-direct',
        '--corrupted_observers', '5', '--corrupted_placement', 'history-optimized-global',
        '--allow_corrupted_region_colocation', '--placement_victim_scope', 'all-locations',
        '--byzantine_reports', '5', '--byzantine_report_mode', 'pair-swap',
        '--background_txs', '300', '--entry_counts', '5', '--training_snapshots', '24',
        '--rho_ms', '0', '--placement_rho_ms', '0', '--disseminations', 'plain',
        '--reporter_policies', 'implementation', '--public_repetitions', '20',
        '--one_way_mode', 'half', '--clock_epsilon_ms', '0', '--seed', '762031'])

class DirectedTrace(p.LatencyTrace):
    def __init__(self,path,delta=0.,network_seed=0):
        super().__init__(path,'half',.15)
        self.delta=delta;self.network_seed=network_seed
        self.factors={}
        for a in self.locations:
            for b in self.locations:
                key=tuple(sorted((a,b)))
                z=p.stable_rng(SEED,'supplement-direction',network_seed,*key).uniform(-1.,1.)
                self.factors[(a,b)]=1. if a==b else 1.+2*delta*z*(1 if a<b else -1)
        if self.is_hourly_snapshot:
            scale=np.array([[self.factors[(a,b)] for b in self.locations] for a in self.locations])
            self.snapshot_matrices=self.snapshot_matrices*scale[None,:,:]

    def sample(self,phase,source,destination,seed,*key):
        value=super().sample(phase,source,destination,seed,*key)
        if self.is_hourly_snapshot:return value
        return value*self.factors[(p.physical_location(source),p.physical_location(destination))]


class Replay:
    def __init__(self,reports,args,ds,dep,trial):
        self.reports=reports;self.threshold=(args.num_nodes-1)//4+1
        self.names=sorted({tx for report in reports for tx,_ in report},key=lambda tx:p.stable_seed(args.seed,ds,dep,trial,'pair-processing-order',tx))
        index={tx:i for i,tx in enumerate(self.names)};self.ti=index['T'];self.fi=index['F']
        ranks=np.empty((len(reports),len(self.names)),np.int32)
        for r,report in enumerate(reports):
            assert len(report)==len(self.names)
            for j,(tx,_) in enumerate(report):ranks[r,index[tx]]=j
        self.before=ranks[:,:,None]<ranks[:,None,:]
        self.earlier=np.arange(len(self.names))[:,None]<np.arange(len(self.names))[None,:]
        self.tapes=[np.array(p.public_permutation(len(self.names),p.stable_seed(args.seed,ds,dep,trial,'public',j))) for j in range(args.public_repetitions)]
        self.inverse=[np.argsort(tape) for tape in self.tapes];self.cache={}

    def evaluate(self,rr,validate=False):
        rr=np.asarray(rr,int);weights=self.before[rr].sum(axis=0)
        adjacency=(weights>=self.threshold)&((weights>weights.T)|((weights==weights.T)&self.earlier))
        strict=int(weights[self.fi,self.ti]>=self.threshold and weights[self.fi,self.ti]>weights[self.ti,self.fi])
        key=adjacency.tobytes()
        if key not in self.cache:
            components=p.tarjan_scc(adjacency);which={v:i for i,c in enumerate(components) for v in c}
            if which[self.ti]!=which[self.fi]:result=float(adjacency[self.fi,self.ti])
            else:
                values=[]
                for tape,inv in zip(self.tapes,self.inverse):
                    perm=adjacency[np.ix_(tape,tape)];ti,fi=int(inv[self.ti]),int(inv[self.fi])
                    component=next(c for c in p.tarjan_scc(perm) if ti in c)
                    order=p.themis_hamiltonian_cycle(component,perm)
                    values.append(order.index(fi)<order.index(ti))
                result=float(np.mean(values))
            self.cache[key]=result
        result=self.cache[key]
        if validate:
            _,_,original=p.build_graph(self.reports,rr,self.threshold,self.names)
            assert np.array_equal(adjacency,original)
            full=np.mean([p.finalize_graph(self.names,original,permutation=tape)['final_ft'] for tape in self.tapes])
            assert abs(result-full)<1e-12
        return dict(strict_edge=strict,final_success=result,joint_success=strict*result,
                    F_votes=int(weights[self.fi,self.ti]),T_votes=int(weights[self.ti,self.fi]))


def report_collection(ready,bad,transport,leader,policy,close=2000.):
    ready=np.asarray(ready,float);bad=np.asarray(bad,int)
    send=np.maximum(close,ready)+.05
    if policy=='leader-ready':send=np.maximum(1000.,ready)+.05
    elif policy=='leader-early-byzantine':send[bad]=ready[bad]+.05
    elif policy!='leader-deadline':raise ValueError(policy)
    arrived=send+np.asarray(transport)
    rr=np.lexsort((np.arange(len(ready)),arrived))[:16]
    return rr,send,arrived


def retrain(trace,dep,args):
    d=dict(dep);bad=d['corrupted_replica_indices'];sites=list(d['replicas'])
    # Original make_deployment trains BEFORE assigning logical aliases.
    # Supply that same physical-location view to the unmodified optimizer.
    draft=dict(d);draft['replicas']=[p.physical_location(site) for site in sites]
    chosen,info=p.optimize_corrupted_locations(trace,draft,bad,args)
    for i,site in zip(bad,chosen):
        # Do not rename any honest identity when a relocated C shares its
        # profile. Identity strings are also part of empirical-packet seeds.
        if p.physical_location(sites[i])!=site:
            sites[i]=p.logical_replica_location(site,i)
    assert len(set(sites))==len(sites)
    assert all(sites[i]==dep['replicas'][i] for i in range(len(sites)) if i not in bad)
    d['replicas']=sites
    d['entries']=p.make_corrupted_direct_sources(d['replicas'],bad)
    d['supplement_placement_training']=info
    return d

def intervals(frame,columns,seed,repetitions=10000):
    collapsed=frame.groupby(['deployment','trial'])[columns].mean()
    deps=sorted(frame.deployment.unique());trials=sorted(frame.trial.unique())
    index=pd.MultiIndex.from_product([deps,trials],names=['deployment','trial'])
    values=collapsed.reindex(index).to_numpy().reshape(len(deps),len(trials),len(columns))
    assert np.isfinite(values).all()
    rng=np.random.default_rng(seed)
    dd=rng.integers(len(deps),size=(repetitions,len(deps),1))
    tt=rng.integers(len(trials),size=(repetitions,len(deps),len(trials)))
    boots=values[dd,tt].mean(axis=(1,2));bounds=np.quantile(boots,[.025,.975],axis=0)
    result={}
    for i,col in enumerate(columns):
        result[col]=float(values[:,:,i].mean())
        result[col+'_lo']=float(bounds[0,i]);result[col+'_hi']=float(bounds[1,i])
    return result

class Samples:
    def __init__(self, path, seed):
        self.seed = seed
        frame = pd.read_csv(path)
        self.locations = sorted(frame.source.unique())
        self.index = {name:i for i,name in enumerate(self.locations)}
        self.samples = np.empty((2,len(self.locations),len(self.locations),750))
        self.counts = np.zeros((2,len(self.locations),len(self.locations)),dtype=int)
        self.mean = np.zeros((len(self.locations),len(self.locations)))
        for (a,b), block in frame.groupby(['source','destination'], sort=False):
            values = block.time.to_numpy()/2.
            for phase in (0,1):
                part = values[phase::2]
                if not 0<len(part)<=750:
                    raise ValueError('unexpected AWS sample count')
                i,j=self.index[a],self.index[b]
                self.samples[phase,i,j,:len(part)] = part
                self.counts[phase,i,j] = len(part)
                if phase==0:self.mean[i,j]=part.mean()
        if not np.all(self.counts):raise ValueError('missing AWS path')

    def rng(self, *key):
        return np.random.default_rng(p.stable_seed(self.seed,'region-gossip',*key))

    def draw(self, phase, sources, destinations, *key, size=None):
        sources, destinations = np.broadcast_arrays(sources,destinations)
        shape = sources.shape if size is None else (size,)+sources.shape
        indices = (self.rng(*key).random(shape)*self.counts[phase,sources,destinations]).astype(int)
        return self.samples[phase,sources,destinations,indices]
