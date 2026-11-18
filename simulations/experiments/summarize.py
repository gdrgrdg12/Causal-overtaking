"""Rebuild paper tables and figures from workload CSVs; no network access."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from common import intervals, p


def summarize_attack(out,repeats=10000):
    out=Path(out)
    paths=sorted((out/'parts').glob('*.csv'))
    data=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True) if paths else pd.read_csv(out/'all_cases.csv.gz')
    data['condition_b']=data.condition_b.fillna(-1).astype(int) if 'condition_b' in data else -1
    keys=['dataset','n','stage','delta','placement','policy','condition_b']
    metrics=['strict_edge','final_success','joint_success','accepted_bad']
    rows=[]
    for labels,g in data.groupby(keys,dropna=False,sort=True):
        r=dict(zip(keys,labels))
        r.update(intervals(g,metrics,p.stable_seed(762031,'supplement-ci',*labels),repeats))
        r['workloads']=len(g[['deployment','trial']].drop_duplicates())
        rows.append(r)
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    for stage,file in [('baseline','baseline_with_ci.csv'),('reporters','reporter_summary.csv'),('asymmetry','asymmetry_summary.csv')]:
        summary[summary.stage==stage].to_csv(out/file,index=False)
    data.to_csv(out/'all_cases.csv.gz',index=False,compression='gzip')
    natural=data[(data.stage=='reporters')&~data.policy.isin(['conditional-b','paper-fixed'])]
    if len(natural):
        dist=natural.groupby(['dataset','policy','accepted_bad']).size().rename('rows').reset_index()
        dist['frequency']=dist['rows']/dist.groupby(['dataset','policy'])['rows'].transform('sum')
        dist.to_csv(out/'reporter_bad_count_distribution.csv',index=False)
    print(summary[['dataset','n','stage','delta','placement','policy','condition_b','strict_edge','final_success']].to_string(index=False))
    return summary


def summarize_entries(out,repeats=10000):
    out=Path(out);paths=sorted((out/'parts').glob('d*/*-cases.csv'))
    a=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
    a.to_csv(out/'cases.csv',index=False)
    singles=pd.concat([pd.read_csv(f) for f in sorted((out/'parts').glob('d*/*-singletons.csv'))],ignore_index=True)
    singles.to_csv(out/'singletons.csv',index=False)
    pools=pd.concat([pd.read_csv(f) for f in sorted((out/'parts').glob('d*/*-full-pool-singleton-support.csv'))],ignore_index=True)
    pools.to_csv(out/'full-pool-singleton-support.csv',index=False)
    metrics=['honest_Ffirst','selected_honest_Ffirst','strict_edge','final_success','joint_success',
             'strict_composition_only','final_gain_over_best_single']
    summary=a.groupby(['dataset','family','K'],as_index=False)[metrics].mean()
    summary.to_csv(out/'summary.csv',index=False)
    ci=[]
    for labels,g in a.groupby(['dataset','family','K']):
        ci.append(dict(zip(['dataset','family','K'],labels))|intervals(g,metrics,p.stable_seed(762031,'entry-ci',*labels),repeats))
    pd.DataFrame(ci).to_csv(out/'bootstrap_summary.csv',index=False)
    print(summary.to_string(index=False))


def summarize_adaptive(out,repeats=10000):
    out=Path(out);a=pd.concat([pd.read_csv(f) for f in sorted((out/'parts').glob('*.csv'))],ignore_index=True)
    a.to_csv(out/'cases.csv',index=False)
    metrics=['guard_final','guard_direct_F','baseline_final','baseline_strict','edge_T_final_F','edge_F_final_F']
    summary=a.groupby(['dataset','policy'],as_index=False)[metrics].mean()
    summary.to_csv(out/'summary.csv',index=False)
    rows=[]
    for ds,sub in a.groupby('dataset'):
        deps=sorted(sub.deployment.unique());trials=sorted(sub.trial.unique());nd,nt=len(deps),len(trials)
        rng=np.random.default_rng(762031+(ds=='wp'))
        d=rng.integers(nd,size=(repeats,nd,1));t=rng.integers(nt,size=(repeats,nd,nt))
        baseline=sub[sub.policy=='fixed-original'].pivot(index='deployment',columns='trial',values='guard_final').reindex(index=deps,columns=trials).to_numpy()
        for policy,g in sub.groupby('policy'):
            x=g.pivot(index='deployment',columns='trial',values='guard_final').reindex(index=deps,columns=trials).to_numpy()
            boot=x[d,t].mean(axis=(1,2));delta=(x-baseline)[d,t].mean(axis=(1,2))
            lo,hi=np.quantile(boot,[.025,.975]);dl,dh=np.quantile(delta,[.025,.975])
            rows.append(dict(dataset=ds,policy=policy,estimate=100*x.mean(),ci_low=100*lo,ci_high=100*hi,
                delta_vs_native=100*(x-baseline).mean(),delta_ci_low=100*dl,delta_ci_high=100*dh))
    pd.DataFrame(rows).to_csv(out/'bootstrap_summary.csv',index=False)
    table=[]
    for policy,label in [('local-fast-actual','Early target relay'),('exchange-fast-actual','Exchange and early relay'),('history-adaptive','Historical policy selection')]:
        g=summary[summary.policy==policy]
        table.append(dict(attack_policy=label,**{r.dataset.upper():100*r.guard_final for r in g.itertuples()}))
    pd.DataFrame(table).to_csv(out/'table8.csv',index=False)
    for kind in ['train','validation']:
        frames=[]
        for path in sorted((out/'training').glob('*-'+kind+'.csv')):
            frame=pd.read_csv(path)
            frame.insert(0,'deployment',int(path.name.split('-')[1][1:]))
            frame.insert(0,'dataset',path.name.split('-')[0]);frames.append(frame)
        pd.concat(frames,ignore_index=True).to_csv(out/(kind+'.csv'),index=False)
    import json
    selections=[]
    for path in sorted((out/'training').glob('*-selection.json')):
        record=json.loads(path.read_text())
        selections.append(dict(dataset=path.name.split('-')[0],deployment=int(path.name.split('-')[1][1:]),**record['selected']))
    pd.DataFrame(selections).to_csv(out/'selected-policies.csv',index=False)
    print(pd.DataFrame(table).to_string(index=False))


def paper_tables(root):
    """Report percentages explicitly; source CSVs retain unrounded fractions."""
    root=Path(root);dest=root/'tables';dest.mkdir(exist_ok=True)
    for family,k_values,number in [('five-observers',[1,5,20],3),('one-observer',[1,5,20],9)]:
        path=root/'entries/summary.csv'
        if path.exists():
            a=pd.read_csv(path);a=a[(a.family==family)&a.K.isin(k_values)].copy()
            a[['strict_edge','final_success']]*=100
            a[['dataset','K','strict_edge','final_success']].to_csv(dest/f'table{number}.csv',index=False)
    path=root/'attack/summary.csv'
    if not path.exists():path=root/'attack/baseline_with_ci.csv'
    if path.exists():
        # Stored reference run has separate stage summaries.
        pieces=[]
        for filename in ['baseline_with_ci.csv','reporter_summary.csv','asymmetry_summary.csv']:
            if (root/'attack'/filename).exists():pieces.append(pd.read_csv(root/'attack'/filename))
        a=pd.concat(pieces,ignore_index=True)
        for number,sub in [(4,a[(a.stage=='reporters')&(((a.policy=='conditional-b')&(a.condition_b==0))|(a.policy=='leader-deadline'))]),
                           (5,a[(a.stage=='asymmetry')|((a.stage=='baseline')&(a.n==21))])]:
            sub=sub.copy()
            for col in ['strict_edge','final_success','final_success_lo','final_success_hi']:
                if col in sub:sub[col]*=100
            sub.to_csv(dest/f'table{number}.csv',index=False)
    path=root/'gossip/summary.csv'
    if path.exists():
        a=pd.read_csv(path);a=a[a['mode'].isin(['plain','p01-b5','p1-b5','p1-b1','adversarial-p1-b1'])].copy()
        a[['strict_edge','final_success']]*=100
        a[['dataset','mode','strict_edge','final_success','gossip_messages','gossip_copies']].to_csv(dest/'table6.csv',index=False)
    path=root/'regional/cases.csv'
    if path.exists():
        a=pd.read_csv(path);rows=[dict(dataset='Either',method='Uniform random',direct_F=np.nan,final_attack=50.,normal_order_loss=50.)]
        for ds,g in a.groupby('dataset'):
            for method,final,normal,direct in [('Themis','baseline_final_ASR','normal_baseline_final_wrong_rate','baseline_direct_F'),
                                               ('Regional ordering','final_ASR','normal_guard_final_wrong_rate','edge_F_probability')]:
                rows.append(dict(dataset=ds,method=method,direct_F=100*g[direct].mean() if direct in g else np.nan,
                                 final_attack=100*g[final].mean(),normal_order_loss=100*g[normal].mean()))
        pd.DataFrame(rows).to_csv(dest/'table7.csv',index=False)
    if (root/'adaptive/table8.csv').exists():
        pd.read_csv(root/'adaptive/table8.csv').to_csv(dest/'table8.csv',index=False)
    print('Paper tables:',dest)


def figures(root):
    import matplotlib.pyplot as plt
    root=Path(root);dest=root/'figures';dest.mkdir(exist_ok=True)
    pdf_metadata={'CreationDate':None,'ModDate':None}
    pth=root/'attack/baseline_with_ci.csv'
    if pth.exists():
        a=pd.read_csv(pth);fig,axes=plt.subplots(1,2,figsize=(7,2.5))
        for ax,metric in zip(axes,['strict_edge','final_success']):
            for ds,g in a.groupby('dataset'):
                g=g.sort_values('n');y=g[metric].to_numpy()*100
                ax.errorbar(g.n,y,yerr=np.array([y-g[metric+'_lo']*100,g[metric+'_hi']*100-y]),marker='o',label=ds.upper())
            ax.set(xlabel='Replicas',ylabel='Rate (%)',ylim=(0,105));ax.legend()
        fig.tight_layout();fig.savefig(dest/'figure2.pdf',metadata=pdf_metadata);plt.close(fig)
    path=root/'capability/summary.csv'
    if path.exists():
        a=pd.read_csv(path);fig,axes=plt.subplots(1,2,figsize=(7,2.5))
        labels={'observers_truthful':'Truthful (c,0)','single_observer_reports':'One observer (1,c)',
                'observers_one_report':'One report (c,1)','joint':'Combined (c,c)'}
        for ax,ds in zip(axes,['aws','wp']):
            for strategy,g in a[a.dataset==ds].groupby('strategy'):
                g=g.sort_values('actual_corruptions');y=g.strict_edge.to_numpy()*100
                lo=g.strict_edge_ci95_low.to_numpy()*100;hi=g.strict_edge_ci95_high.to_numpy()*100
                ax.errorbar(g.actual_corruptions,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),marker='o',label=labels[strategy])
            ax.set(title=ds.upper(),xlabel='Actual corruptions c',ylabel='Strict-edge rate (%)',ylim=(0,105))
        axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(dest/'figure3.pdf',metadata=pdf_metadata);plt.close(fig)
    paths=sorted((root/'rho').glob('*/*/summary.csv'))
    if paths:
        a=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        a=a[(a.arm=='full_union')&(a.dissemination=='plain')]
        a.to_csv(root/'rho/summary.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(7,2.5))
        for ax,prefix in zip(axes,['strict_direct_edge_FT','final_F_before_T_rate']):
            for ds,g in a.groupby('dataset'):
                g=g.sort_values('rho_ms');y=g[prefix+'_mean'].to_numpy()*100
                lo=g[prefix+'_ci95_low'].to_numpy()*100;hi=g[prefix+'_ci95_high'].to_numpy()*100
                ax.errorbar(g.rho_ms,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),marker='o',label=ds.upper())
            ax.set(xlabel='Response delay (ms)',ylabel='Rate (%)',ylim=(0,105));ax.legend()
        fig.tight_layout();fig.savefig(dest/'figure4.pdf',metadata=pdf_metadata);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--plots',action='store_true');cfg=parser.parse_args()
    paper_tables(cfg.root)
    if cfg.plots:figures(cfg.root)
