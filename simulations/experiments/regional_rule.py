"""Observed two-copy precedence, regional sampling, original SCC/Unspool."""
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

from regional_support import assert_tournament, ft_final

class ObservedPrecedenceEvidence:
    def __init__(self,direct,peer,ranks,groups,core,common):
        self.n,self.m=direct.shape;self.groups=groups;self.core=np.asarray(core)
        self.common=common;self.available=common.any(axis=2);self.ranks=ranks
        self.before=ranks[:,:,None]<ranks[:,None,:]
        self.order_pool=common.copy();self.order_pool[self.core]=True
        lo=np.minimum(direct,peer);hi=np.maximum(direct,peer)
        # A tiny floating-point equality tolerance, not a learned time margin.
        stable=(lo[:,None,:]-hi[:,:,None]>1e-9)&common
        self.region=np.array([stable[groups==g].sum(axis=0)>=int(np.sum(groups==g))//2+1
                              for g in np.unique(groups)])

    def complete_graph(self,base,priorities,tie_ranks):
        votes=np.zeros_like(base,dtype=np.int8)
        for g in np.unique(self.groups):
            ids=np.asarray(priorities[int(g)]);eligible=self.order_pool[ids]
            selected=eligible&(np.cumsum(eligible,axis=0)<=3)
            k=selected.sum(axis=0);left=(selected&self.before[ids]).sum(axis=0)
            rank=tie_ranks[int(g)];tie=rank[:,None]<rank[None,:]
            prefer=(2*left>k)|((2*left==k)&tie)
            fixed=self.region[int(g)]|self.region[int(g)].T
            prefer[fixed]=self.region[int(g)][fixed]
            votes+=prefer
        result=votes>=3
        np.fill_diagonal(result,False)
        return result

class ConditionalTFDraw:
    """Enumerate tiny per-region source-priority spaces, not full graph draws.

    First condition on the exact TF edge outcome, then sample all source
    priorities and region tie ranks from their correct conditional law. Mix
    both outcomes using their exact probabilities, without changing attacks.
    """
    def __init__(self,e,ti,fi):
        self.e=e;self.ti=ti;self.fi=fi;self.options=[];ps=[]
        for g in np.unique(e.groups):
            choices=[[],[]]
            for perm in permutations(np.flatnonzero(e.groups==g).tolist()):
                selected=[i for i in perm if e.order_pool[i,ti,fi]][:3]
                k=len(selected);fv=sum(e.ranks[i,fi]<e.ranks[i,ti] for i in selected)
                for coin in (0,1):
                    f=(2*fv>k) or (2*fv==k and coin==1)
                    if e.region[int(g),ti,fi]:f=False
                    if e.region[int(g),fi,ti]:f=True
                    choices[int(f)].append((perm,coin))
            self.options.append(choices);ps.append(len(choices[1])/sum(map(len,choices)))
        self.ps=np.array(ps);self.patterns=[];self.weights=[]
        for bits in product((0,1),repeat=5):
            weight=float(np.prod(np.where(bits,self.ps,1-self.ps)))
            if weight:self.patterns.append(bits);self.weights.append(weight)
        self.patterns=np.array(self.patterns,int);self.weights=np.array(self.weights)
        self.outcomes=(self.patterns.sum(axis=1)>=3).astype(int)
        self.pf=float(self.weights[self.outcomes==1].sum())

    def draw(self,want_f,rng,m):
        allowed=np.flatnonzero(self.outcomes==int(want_f))
        weights=self.weights[allowed];weights=weights/weights.sum()
        bits=self.patterns[int(rng.choice(allowed,p=weights))]
        priorities=[];tie_ranks=[]
        for g,bit in enumerate(bits):
            options=self.options[g][int(bit)];perm,coin=options[int(rng.integers(len(options)))]
            rank=np.argsort(rng.permutation(m))
            if int(rank[self.fi]<rank[self.ti])!=coin:
                rank[self.ti],rank[self.fi]=rank[self.fi],rank[self.ti]
            priorities.append(np.array(perm));tie_ranks.append(rank)
        return priorities,tie_ranks

def outcome(e,base,tapes,ds,dep,trial,ti,fi,validate=False):
    draw=ConditionalTFDraw(e,ti,fi);final=reverse=follow=0.
    for rep,tape in enumerate(tapes):
        for want,prob in ((0,1-draw.pf),(1,draw.pf)):
            if prob<1e-14:continue
            rng=np.random.default_rng(p.stable_seed(762031,'old400-full-conditional',ds,dep,trial,rep,want))
            priorities,ties=draw.draw(want,rng,len(base))
            adj=e.complete_graph(base,priorities,ties);assert_tournament(adj)
            assert int(adj[fi,ti])==want
            value=ft_final(adj,tape,ti,fi,validate and rep==0)
            final+=prob*value/len(tapes);reverse+=prob*value*(1-want)/len(tapes)
            follow+=prob*value*want/len(tapes)
    return dict(direct_F=draw.pf,final_attack=final,edge_T_final_F=reverse,edge_F_final_F=follow)
