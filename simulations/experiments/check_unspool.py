"""Check Python replay against the original C++ Unspool implementation."""
from pathlib import Path
import argparse
import subprocess
from common import p, args_for, Replay
from run_original_unspool_conformance import materialize, TOTAL_PATTERN


def cpp_check(root,repo):
    dest=root/'cpp-check';dest.mkdir(exist_ok=True)
    binary=(dest/'unspool').resolve()
    subprocess.run(['c++','-std=c++17','-O3','-I'+str(repo/'simulations/include'),
        str(repo/'simulations/src-unspooling_algo/unspool.cpp'),'-o',str(binary)],check=True)
    for ds in ['aws','wp']:
        args=args_for(ds,repo);trace=p.LatencyTrace(args.ping_files[0],'half',.15)
        d=p.make_deployment(trace,0,args);snap=p.trial_snapshot(trace,d,0,args)
        f=p.union_arrivals(snap['singleton_arrivals'],[e['id'] for e in d['entries']])
        physical=p.apply_dissemination(trace,d,0,snap['direct_without_f'],f,'plain',args)
        reports,_=p.apply_byzantine_reports([physical[r] for r in d['replicas']],d,args)
        r=Replay(reports,args,ds,0,0);rr=d['reporter_order'][:16]
        expected=r.evaluate(rr,validate=True)['final_success']
        case=dest/(ds+'.case');p.write_conformance_case(str(case),reports,rr,r.names,r.tapes)
        paths=materialize(str(case),str(dest/'materialized'))
        result=subprocess.run([str(binary),paths[0],'--fixed-quorum-file',paths[1],
            '--pair-order-file',paths[2],'--public-order-file',paths[3]],capture_output=True,text=True,check=True)
        match=TOTAL_PATTERN.search(result.stdout)
        if not match:raise AssertionError('No C++ result')
        yes,total=map(int,match.groups());assert abs(yes/total-expected)<1e-12
        print(ds,'original C++ replay matches',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    cfg=parser.parse_args()
    cpp_check(cfg.root,Path(__file__).resolve().parents[2])


if __name__=='__main__':main()
