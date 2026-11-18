"""Measured regional propagation, receipt collection, and causal attack policies."""
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

from common import Samples
matched=p
_SAMPLES={}

def geo_distances(latlon):
    p=np.radians(np.asarray(latlon,float))
    a=np.sin((p[:,None,0]-p[None,:,0])/2)**2+np.cos(p[:,None,0])*np.cos(p[None,:,0])*np.sin((p[:,None,1]-p[None,:,1])/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0.,1.)))


class WPSamples:
    def __init__(self,repo,seed):
        self.seed=seed
        with np.load(repo/'simulations/data-ping/wp-hourly-2020-07-19-20.npz',allow_pickle=False) as a:
            self.locations=a['locations'].astype(str).tolist()
            self.hours=a['hours'].astype(str).tolist()
            self.matrices=a['avg_owd'].astype(float)
        self.index={x:i for i,x in enumerate(self.locations)}
        h=self.matrices[::2]
        self.mean=h.mean(axis=0);self.lower=h.min(axis=0);self.upper=h.max(axis=0)
        coords=pd.read_csv(repo/'simulations/data-ping/wp-servers-2020-07-19.csv',dtype={'id':str}).set_index('id').reindex(self.locations)
        self.latlon=coords[['latitude','longitude']].to_numpy(float)
        self.cities=coords.title.fillna(pd.Series(self.locations,index=self.locations)).astype(str).tolist()
        self.mapped=np.isfinite(self.latlon).all(axis=1)
        self.geo=geo_distances(self.latlon)
        self.active=1

    def rng(self,*key):
        return np.random.default_rng(matched.stable_seed(self.seed,'region-gossip','wp',*key))

    def set_case(self,dep,trial):
        self.active=int(self.rng('execution-hour',dep,trial).choice(np.arange(1,48,2)))

    def draw(self,phase,sources,destinations,*key,size=None):
        sources,destinations=np.broadcast_arrays(sources,destinations)
        if phase!=1:raise ValueError('history is frozen explicitly,not drawn here')
        a=self.matrices[self.active,sources,destinations]
        return a.copy() if size is None else np.broadcast_to(a,(size,)+a.shape)


def samples_for(dataset,repo):
    if dataset not in _SAMPLES:
        if dataset=='aws':
            s=Samples(repo/'simulations/data-ping/aws-individual_pings.csv',762031)
            s.lower=np.array([[s.samples[0,i,j,:s.counts[0,i,j]].min() for j in range(21)] for i in range(21)])
            s.upper=np.array([[s.samples[0,i,j,:s.counts[0,i,j]].max() for j in range(21)] for i in range(21)])
        else:s=WPSamples(repo,762031)
        _SAMPLES[dataset]=s
    return _SAMPLES[dataset]

def disseminate(direct, links, groups, forwarders, processing_ms=.05):
    """First receipt and first peer copy in an eager, within-group network.

    links[...,sender,recipient] is sampled separately for each transaction.
    Only forwarders retransmit; all nodes, including silent faulty observers,
    may receive earlier through an honest peer. Each forwards its first copy.
    """
    direct = np.asarray(direct, dtype=float)
    groups = np.asarray(groups)
    n = direct.shape[-1]
    processing=np.asarray(processing_ms,dtype=float)
    if processing.ndim and processing.shape!=(n,):
        raise ValueError('processing must be scalar or one cost per sender')
    if np.any(np.asarray(links) < 0) or np.any(processing < 0):
        raise ValueError('nonnegative transmission and processing required')
    allowed = (groups[:, None] == groups[None, :]) & ~np.eye(n, dtype=bool)
    allowed &= np.asarray(forwarders, dtype=bool)[:, None]
    cost=processing if processing.ndim==0 else processing[:,None]
    edges = np.where(allowed, np.asarray(links) + cost, np.inf)
    receipt = direct.copy()
    for _ in range(n):
        forwarded = np.min(receipt[..., :, None] + edges, axis=-2)
        updated = np.minimum(direct, forwarded)
        if np.array_equal(receipt, updated):
            return receipt, forwarded
        receipt = updated
    raise AssertionError('positive group forwarding did not converge')


def reported_channels(first, gap):
    """Represent a claimed first receipt and signed direct-minus-peer gap."""
    first, gap = np.asarray(first), np.asarray(gap)
    return first + np.maximum(gap, 0.), first + np.maximum(-gap, 0.)

def complete_reports(background,t,f):
    return [sorted([(f'B{i:03d}',float(background[i,r])) for i in range(len(background))]+
                   [('T',float(t[r])),('F',float(f[r]))],key=lambda pair:pair[1]) for r in range(len(t))]

def public_inputs(dataset,dep,trial,count,repetitions,seed,phase):
    names=sorted([f'B{i:03d}' for i in range(count)]+['T','F'],key=lambda tx:
        p.stable_seed(seed,dataset,dep,trial,'hybrid-pair-order',phase,tx))
    tapes=[p.public_permutation(len(names),p.stable_seed(
        seed,dataset,dep,trial,'hybrid-public',phase,r)) for r in range(repetitions)]
    return names,tapes

def full_original_order(adj,tape):
    """Same original SCC/Hamiltonian finalizer, also return ALL positions.

    These graphs remain tournaments; SCC condensation is a total order, so
    reversing Tarjan's sink-first emission is the original unique DAG order.
    """
    perm=adj[np.ix_(tape,tape)]
    components=p.tarjan_scc(perm)
    order=np.array([int(tape[x]) for c in reversed(components)
                    for x in p.themis_hamiltonian_cycle(c,perm)],int)
    assert sorted(order.tolist())==list(range(len(adj)))
    which=np.zeros(len(adj),int)
    for k,c in enumerate(components):which[tape[c]]=k
    return order,which,[len(c) for c in components]


def assert_tournament(adj):
    m=len(adj)
    assert not np.any(np.diag(adj))
    assert not np.any(adj&adj.T)
    assert np.all((adj|adj.T)|np.eye(m,dtype=bool))


def ranks_from_reports(reports,names):
    index={name:i for i,name in enumerate(names)}
    ranks=np.zeros((21,len(names)),int);stamps=np.zeros_like(ranks,dtype=float)
    for i,row in enumerate(reports):
        for rank,(name,stamp) in enumerate(row):
            ranks[i,index[name]]=rank;stamps[i,index[name]]=stamp
    return ranks,stamps

def ft_final(adj,tape,ti,fi,validate=False):
    """Same original SCC/Hamiltonian relative order, skip unrelated SCCs."""
    perm=adj[np.ix_(tape,tape)];inv=np.argsort(tape)
    pti,pfi=int(inv[ti]),int(inv[fi])
    components=p.tarjan_scc(perm)
    tc=next(c for c in components if pti in c)
    if pfi not in tc:
        outcome=int(adj[fi,ti])
    else:
        order=p.themis_hamiltonian_cycle(tc,perm)
        outcome=int(order.index(pfi)<order.index(pti))
    if validate:
        order,_,_=full_original_order(adj,tape);position=np.argsort(order)
        assert outcome==int(position[fi]<position[ti])
    return outcome

def f_receipt(observed,reaction,bad,sites,oe,er,catalog,count):
    launch=observed[bad]+reaction
    direct=(launch[:,None]+er[np.arange(5),sites[bad],:]).min(axis=0)
    if count:
        chosen=catalog[:count]
        via=((launch[:,None]+oe[:,chosen])[:,:,None]+er[:,chosen,:]).min(axis=(0,1))
        direct=np.minimum(direct,via)
    direct[bad]=np.minimum(direct[bad],launch)
    return direct


def gather_records(groups, honest, core, dt,gt,df,gf, delivery, close=2000.):
    ready = np.maximum.reduce([dt,gt,df,gf])+.05
    complete = set(int(i) for i in core if ready[i]<=close)
    for carrier in core:
        for sender in np.flatnonzero(groups==groups[carrier]):
            if ready[sender]+delivery[sender,carrier]>close: continue
            # Corrupt carriers can select among the local messages they have
            # actually received; they never read unavailable global H times.
            supports_f = df[sender]<dt[sender] and gf[sender]<gt[sender]
            if honest[carrier] or supports_f: complete.add(int(sender))
    channel_ids = np.array(sorted(complete),dtype=int)
    order_ids = np.array(sorted(set(int(i) for i in core)|complete),dtype=int)
    return channel_ids, order_ids

def witness_verdict(dt,gt,df,gf,groups,rr,jitter,kind):
    regional=[]
    for group in np.unique(groups):
        members=np.asarray(rr)[groups[np.asarray(rr)]==group]
        q=int(np.sum(groups==group))//2+1
        if kind=='direct-only':
            t=(df[members]-dt[members])>1e-9
            f=(dt[members]-df[members])>1e-9
        elif kind=='two-channel-order':
            t=((df[members]-dt[members])>1e-9)&((gf[members]-gt[members])>jitter[members]+1e-9)
            f=((dt[members]-df[members])>1e-9)&((gt[members]-gf[members])>jitter[members]+1e-9)
        elif kind=='two-copy-disjoint':
            t=np.minimum(df[members],gf[members])-np.maximum(dt[members],gt[members])>jitter[members]+1e-9
            f=np.minimum(dt[members],gt[members])-np.maximum(df[members],gf[members])>jitter[members]+1e-9
        else:raise ValueError(kind)
        regional.append('T-first' if int(t.sum())>=q else 'F-first' if int(f.sum())>=q else None)
    # Evidence from3 of5 distinct regions; no actual bad identities enter.
    verdict='T-first' if regional.count('T-first')>=3 else 'F-first' if regional.count('F-first')>=3 else None
    return verdict,regional

def final_f_probability(probabilities):
    distribution=np.array([1.])
    for p in probabilities:
        distribution=np.convolve(distribution,[1.-p,p])
    n=len(probabilities)
    if n%2!=1:raise ValueError('pilot uses five nonempty regional votes')
    return float(distribution[n//2+1:].sum())

def report_probability(rt,rf,groups,reporters,certs,draws):
    rr=np.asarray(reporters);probabilities=[]
    for group,cert in zip(np.unique(groups),certs):
        if cert=='T':probabilities.append(1.);continue
        if cert=='F':probabilities.append(0.);continue
        members=rr[groups[rr]==group];n=len(members)
        if not n:probabilities.append(.5);continue
        f=int(np.sum(rf[members]<rt[members]))
        if draws==1:probabilities.append(f/n)
        else:
            k=min(draws,n);den=comb(n,k);p=0.
            for j in range(k+1):
                if j>f or k-j>n-f:continue
                weight=1. if 2*j>k else .5 if 2*j==k else 0.
                p+=weight*comb(f,j)*comb(n-f,k-j)/den
            probabilities.append(p)
    return final_f_probability(probabilities)

def receipt_availability(direct,peer,ranks,groups,core,honest,delivery,close=2000.):
    """Exact old gather_records generalized to every pair, including omission.

    A faulty carrier forwards a sender's pair record only when both claimed
    channels agree with the carrier's own reported pair direction. On TF this
    is EXACTLY old supports_f withholding; no unseen H record is consulted.
    """
    ready=np.maximum(direct,peer)+.05
    m=direct.shape[1];common=np.zeros((21,m,m),bool)
    for i in core:common[i]=(ready[i,:,None]<=close)&(ready[i,None,:]<=close)
    for carrier in core:
        cb=ranks[carrier,:,None]<ranks[carrier,None,:]
        for sender in np.flatnonzero(groups==groups[carrier]):
            arrived=ready[sender]+delivery[sender,carrier]<=close
            pair=arrived[:,None]&arrived[None,:]
            if not honest[carrier]:
                sb=(direct[sender,:,None]<direct[sender,None,:])&(peer[sender,:,None]<peer[sender,None,:])
                pair&=(sb&cb)|(sb.T&cb.T)
            common[sender]|=pair
    assert np.array_equal(common,common.transpose(0,2,1))
    return common


def make_case(s,ds,dep,trial,info):
    sites=np.array(info['site_ids']);groups=np.array(info['group_ids']);bad=np.array(info['bad_ids'])
    honest=~np.isin(np.arange(21),bad);core=np.array(info['core']);N=len(s.locations)
    clients=s.rng('clients',dep).choice(N,15,replace=False)
    if ds=='wp':s.set_case(dep,trial)
    key=(dep,trial);victim=int(s.rng('victim',*key).choice(clients))
    dt=500.+s.draw(1,victim,sites,'T',*key)
    tl=s.draw(1,sites[:,None],sites[None,:],'T-links',*key)
    fl=s.draw(1,sites[:,None],sites[None,:],'F-links',*key)
    oe=s.draw(1,sites[bad,None],np.arange(N)[None,:],'observer-entry',*key)
    er=s.draw(1,np.arange(N)[None,:,None]+np.zeros((5,1,1),int),sites[None,None,:],'entry-recipient',*key)
    t,pt=disseminate(dt,tl,groups,honest,.05)
    rng=s.rng('background',*key);sources=np.resize(clients,300);rng.shuffle(sources)
    releases=rng.exponential(1000./300,300).cumsum()
    bgd=releases[:,None]+s.draw(1,sources[:,None],sites[None,:],'background-direct',*key)
    bl=s.draw(1,sites[:,None],sites[None,:],'background-links',*key,size=300)
    bg,pbg=disseminate(bgd,bl,groups,honest,.05)
    delivery=s.draw(1,sites[:,None],sites[None,:],'regional-witness-delivery',dep,trial)
    df=f_receipt(t,0.,bad,sites,oe,er,np.arange(N),0)
    f,pf=disseminate(df,fl,groups,honest,.05)
    names,tapes=public_inputs(ds,dep,trial,300,20,762031,'region-gossip')
    low,high=(np.array(info[k]) for k in ('low','high'));jitter=np.zeros(21)
    rt,rf=t.copy(),f.copy();rt[bad]=np.maximum(t[bad],f[bad])+20.
    native=[f'B{i:03d}' for i in range(300)]+['T','F'];take=[native.index(name) for name in names]
    direct=np.column_stack([bgd.T,dt,df])[:,take]
    peer=np.column_stack([pbg.T,pt,pf])[:,take]
    first=np.column_stack([bg.T,t,f])[:,take]
    reports=complete_reports(bg,rt,rf);ranks,stamps=ranks_from_reports(reports,names)
    for i in bad:direct[i],peer[i]=reported_channels(stamps[i],np.full(302,(low[i]+high[i])/2))
    return dict(groups=groups,bad=bad,honest=honest,core=core,names=names,tapes=[np.array(x) for x in tapes],
        direct=direct,peer=peer,first=first,ranks=ranks,reports=reports,
        low=low,high=high,jitter=jitter,delivery=delivery)

class HistorySamples:
    def __init__(self,parent,ds,fold):
        self.parent=parent;self.ds=ds;self.fold=fold
        self.locations=parent.locations;self.active=0

    def rng(self,*key):
        if key[0]=='clients':return self.parent.rng(*key)
        return self.parent.rng('adaptive-history',self.fold,*key)

    def set_case(self,dep,trial):
        self.active=int(self.rng('history-hour',dep,trial).choice(np.arange(2*self.fold,48,4)))

    def draw(self,phase,sources,destinations,*key,size=None):
        sources,destinations=np.broadcast_arrays(sources,destinations)
        if self.ds=='wp':
            values=self.parent.matrices[self.active,sources,destinations]
            return values.copy() if size is None else np.broadcast_to(values,(size,)+values.shape)
        counts=(self.parent.counts[0,sources,destinations]+1-self.fold)//2
        shape=sources.shape if size is None else (size,)+sources.shape
        indices=(self.rng(*key).random(shape)*counts).astype(int)*2+self.fold
        return self.parent.samples[0,sources,destinations,indices]


def inputs(samples,ds,dep,trial,info):
    c=make_case(samples,ds,dep,trial,info)
    sites=np.array(info['site_ids']);n=len(samples.locations);key=(dep,trial)
    clients=samples.rng('clients',dep).choice(n,15,replace=False)
    victim=int(samples.rng('victim',*key).choice(clients))
    c['raw_t_direct']=500.+samples.draw(1,victim,sites,'T',*key)
    c['raw_t_links']=samples.draw(1,sites[:,None],sites[None,:],'T-links',*key)
    c['raw_f_links']=samples.draw(1,sites[:,None],sites[None,:],'F-links',*key)
    c['raw_f_entry']=samples.draw(1,np.arange(n)[None,:,None]+np.zeros((5,1,1),int),sites[None,None,:],'entry-recipient',*key)
    c['sites']=sites
    return c


def physics(c,policy):
    dt=c['raw_t_direct'];links=c['raw_t_links'];groups=c['groups'];honest=c['honest'];bad=c['bad']
    local=(groups[:,None]==groups[None,:])&~np.eye(21,dtype=bool)
    senders=honest.copy()
    if policy['relay']!='withhold':senders[bad]=True
    allowed=local&senders[:,None]
    if policy['relay']=='exchange-fast':
        allowed[np.ix_(bad,bad)]=~np.eye(5,dtype=bool)
    processing=np.where(honest,.05,0.)
    cost=np.where(allowed,links+processing[:,None],np.inf)
    first=dt.copy()
    for _ in range(21):
        updated=np.minimum(dt,np.min(first[:,None]+cost,axis=0))
        if np.array_equal(updated,first):break
        first=updated
    else:raise AssertionError('T forwarding did not converge')
    peer=np.min(first[:,None]+np.where(local&senders[:,None],links+processing[:,None],np.inf),axis=0)
    df=f_receipt(first,policy['wait'],bad,c['sites'],None,c['raw_f_entry'],None,0)
    f,pf=disseminate(df,c['raw_f_links'],groups,honest,.05)
    rt=first.copy();rf=f.copy();rt[bad]=np.maximum(first[bad],f[bad])+policy['late']
    return first,peer,df,f,pf,rt,rf


def rebuild(c,policy):
    t,pt,df,f,pf,rt,rf=physics(c,policy)
    result=dict(c);direct=c['direct'].copy();peer=c['peer'].copy();first=c['first'].copy()
    ti,fi=c['names'].index('T'),c['names'].index('F')
    direct[:,ti]=c['raw_t_direct'];peer[:,ti]=pt;first[:,ti]=t
    direct[:,fi]=df;peer[:,fi]=pf;first[:,fi]=f
    bg_indices=[c['names'].index(f'B{i:03d}') for i in range(300)]
    reports=complete_reports(c['first'][:,bg_indices].T,rt,rf)
    ranks,stamps=ranks_from_reports(reports,c['names'])
    for i in c['bad']:
        direct[i],peer[i]=reported_channels(stamps[i],np.full(302,(c['low'][i]+c['high'][i])/2))
    result.update(direct=direct,peer=peer,first=first,reports=reports,ranks=ranks)
    return result


def key(policy):return json.dumps(policy,sort_keys=True)


BASE=dict(relay="withhold",wait=0.,late=20.,frame=False)
