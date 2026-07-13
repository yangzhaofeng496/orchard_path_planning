"""Evaluation and ablation for the SE(2) Hybrid A* + risk-corridor MPPI Ours."""
import argparse,csv,os,time
import numpy as np
from benchmark import make_scene,Scene
from se2_hybrid_astar import SE2HybridAStar
from risk_corridor_mppi import RiskCorridorMPPI

def run(n,out):
    os.makedirs(out,exist_ok=True);rows=[]
    variants=[('SE2-Risk-HA*+MPPI (Ours)',True,True,True),('no_global_risk',False,True,True),('no_dynamic_prediction',True,True,False),('no_kinodynamic_recovery',True,False,True)]
    for ep in range(n):
        base=make_scene(4000+ep)
        for name,use_risk,recovery,dynamic in variants:
            s=base if use_risk else Scene(base.occ,np.zeros_like(base.risk),base.rows,base.start,base.goal,base.dyn)
            planner=SE2HybridAStar();t=time.perf_counter();corridor,gi=planner.plan(s,max_expansions=30000);global_ms=(time.perf_counter()-t)*1000
            if corridor:
                t=time.perf_counter();traj,li=RiskCorridorMPPI(samples=24,horizon=10,seed=ep).track(base,corridor,max_steps=700,enable_recovery=recovery,use_dynamic=dynamic);local_ms=(time.perf_counter()-t)*1000
            else:traj=np.empty((0,3));li={'success':False,'collisions':0,'recoveries':0};local_ms=0
            length=float(np.linalg.norm(np.diff(traj[:,:2],axis=0),axis=1).sum()) if len(traj)>1 else 0
            rows.append(dict(episode=ep,method=name,success=int(li['success']),collisions=li['collisions'],recoveries=li['recoveries'],path_length=length,global_ms=global_ms,local_ms=local_ms,expanded=gi.get('expanded',0)))
            print(ep,name,rows[-1]['success'],rows[-1]['collisions'])
    with open(os.path.join(out,'two_layer_metrics.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for name,*_ in variants:
        z=[r for r in rows if r['method']==name];summary.append(dict(method=name,episodes=len(z),success_rate=np.mean([r['success'] for r in z]),mean_collisions=np.mean([r['collisions'] for r in z]),mean_length=np.mean([r['path_length'] for r in z]),mean_global_ms=np.mean([r['global_ms'] for r in z]),mean_local_ms=np.mean([r['local_ms'] for r in z]),mean_expanded=np.mean([r['expanded'] for r in z])))
    with open(os.path.join(out,'two_layer_summary.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(os.path.join(out,'two_layer_summary.csv'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=12);p.add_argument('--out',default='results');a=p.parse_args();run(a.episodes,a.out)
