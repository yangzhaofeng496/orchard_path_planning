"""Evaluation and ablation for the SE(2) Hybrid A* + risk-corridor MPPI Ours."""
import argparse,csv,os,time
import numpy as np
from benchmark import make_scene,Scene
from se2_hybrid_astar import SE2HybridAStar
from risk_corridor_mppi import RiskCorridorMPPI
from ackermann_model import AckermannModel

def corridor_tracker(scene,corridor,smooth=False,max_steps=900):
    model=AckermannModel(dt=.1);guard=RiskCorridorMPPI(model=model,samples=4,horizon=4);state=np.array([corridor[0].x,corridor[0].y,corridor[0].yaw]);traj=[state.copy()];last_delta=0.;collisions=0
    for _ in range(max_steps):
        idx=guard.nearest(corridor,state);target=corridor[min(idx+(5 if smooth else 3),len(corridor)-1)]
        desired=np.arctan2(target.y-state[1],target.x-state[0]);err=model.normalize_angle(desired-state[2]);delta=np.clip(np.arctan2(2*model.L*np.sin(err),max(np.hypot(target.x-state[0],target.y-state[1]),.5)),-model.max_steer,model.max_steer)
        if smooth:delta=.75*last_delta+.25*delta
        v=.75 if smooth else .9;proposed=model.update(state,v,delta)
        if guard.collision(scene,proposed):
            safe=[]
            for vv,dd in model.get_motion_primitives():
                q=model.update(state,vv,dd)
                if not guard.collision(scene,q):safe.append((guard.nearest(corridor,q),vv,q,dd))
            if not safe:break
            _,v,proposed,delta=max(safe,key=lambda z:(z[0],z[1]))
        state=proposed;last_delta=delta;collisions+=int(guard.collision(scene,state));traj.append(state.copy())
        if np.hypot(state[0]-corridor[-1].x,state[1]-corridor[-1].y)<.7:break
    success=int(np.hypot(state[0]-corridor[-1].x,state[1]-corridor[-1].y)<.7 and collisions==0)
    return np.asarray(traj),{'success':success,'collisions':collisions,'recoveries':0}

def run(n,out):
    os.makedirs(out,exist_ok=True);rows=[]
    variants=[('SE2 Hybrid A*+Pure Pursuit',False,False,False,'pp'),('SE2 Hybrid A*+TEB-like',False,False,False,'teb'),('SE2-Risk-HA*+MPPI (Ours)',True,True,True,'mppi'),('no_global_risk',False,True,True,'mppi'),('no_dynamic_prediction',True,True,False,'mppi'),('no_kinodynamic_recovery',True,False,True,'mppi')]
    for ep in range(n):
        difficulty=['simple','moderate','hard'][ep%3]
        base=make_scene(4000+ep,difficulty=difficulty)
        for name,use_risk,recovery,dynamic,controller in variants:
            s=base if use_risk else Scene(base.occ,np.zeros_like(base.risk),base.rows,base.start,base.goal,base.dyn)
            planner=SE2HybridAStar();t=time.perf_counter();corridor,gi=planner.plan(s,max_expansions=30000);global_ms=(time.perf_counter()-t)*1000
            if corridor:
                t=time.perf_counter()
                if controller=='pp':traj,li=corridor_tracker(base,corridor,False)
                elif controller=='teb':traj,li=corridor_tracker(base,corridor,True)
                else:traj,li=RiskCorridorMPPI(samples=20,horizon=9,seed=ep).track(base,corridor,max_steps=700,enable_recovery=recovery,use_dynamic=dynamic)
                local_ms=(time.perf_counter()-t)*1000
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
