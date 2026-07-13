"""Ackermann/footprint-aware audit benchmark for the orchard planners."""
from __future__ import annotations
import argparse,csv,math,os,time
import numpy as np
from ackermann_model import AckermannModel
from benchmark import Scene,make_scene,astar,hybrid_astar,greedy_global,rrt_star,local_plan,teb_inspired_plan,reactive_plan,orchard_ua_plan,dist

RES=0.40
VEH=AckermannModel(dt=.1)

def inflate_scene(scene,radius=2):
    """Minkowski-style obstacle inflation shared by every planner."""
    occ=scene.occ.copy(); src=np.argwhere(scene.occ)
    for x,y in src:
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                if dx*dx+dy*dy<=radius*radius:
                    xx,yy=x+dx,y+dy
                    if 0<=xx<occ.shape[0] and 0<=yy<occ.shape[1]:occ[xx,yy]=1
    for cx,cy in (scene.start,scene.goal):
        occ[max(0,cx-radius):cx+radius+1,max(0,cy-radius):cy+radius+1]=False
    return Scene(occ,scene.risk,scene.rows,scene.start,scene.goal,scene.dyn)

def plan(scene,name):
    t=time.perf_counter()
    if name=='A*+DWA': g=astar(scene,False,False); p=local_plan(scene,g,False,True)
    elif name=='Hybrid A*+TEB': g=hybrid_astar(scene); p=teb_inspired_plan(scene,g)
    elif name=='Greedy+APF': g=greedy_global(scene); p=reactive_plan(scene,g,'apf')
    elif name=='RCRA*+DWA': g=astar(scene,True,True); p=local_plan(scene,g,False,True)
    elif name=='A*+UA': g=astar(scene,False,False); p=orchard_ua_plan(scene,g)
    else: g=astar(scene,True,True); p=orchard_ua_plan(scene,g)
    return g,p,(time.perf_counter()-t)*1000

def footprint_collision(scene,state):
    corners=VEH.get_corners(state)/RES
    pts=[state[:2]/RES]
    for a,b in zip(corners,np.roll(corners,-1,axis=0)):
        for u in np.linspace(0,1,8): pts.append(a*(1-u)+b*u)
    for x,y in pts:
        ix,iy=int(round(x)),int(round(y))
        if ix<0 or iy<0 or ix>=90 or iy>=90 or scene.occ[ix,iy]: return True
    return False

def track(scene,path):
    if len(path)<2:return np.array([[scene.start[0]*RES,scene.start[1]*RES,0]]),[],1
    wp=np.asarray(path,float)*RES; d=wp[1]-wp[0]; state=np.array([*wp[0],math.atan2(d[1],d[0])]); states=[state.copy()]; controls=[]; collisions=0; target=1
    for _ in range(1800):
        while target<len(wp)-1 and np.linalg.norm(wp[target]-state[:2])<.45: target+=1
        vec=wp[target]-state[:2]; desired=math.atan2(vec[1],vec[0]); err=VEH.normalize_angle(desired-state[2])
        delta=np.clip(math.atan2(2*VEH.L*math.sin(err),max(np.linalg.norm(vec),.5)),-VEH.max_steer,VEH.max_steer)
        v=.9*max(.25,1-abs(delta)/VEH.max_steer); state=VEH.update(state,v,delta); states.append(state.copy());controls.append((v,delta))
        collisions+=int(footprint_collision(scene,state))
        if np.linalg.norm(state[:2]-wp[-1])<.65:break
    return np.asarray(states),controls,collisions

def eval_one(scene,states,controls,planning_ms,collisions):
    xy=states[:,:2]; seg=np.linalg.norm(np.diff(xy,axis=0),axis=1); occ=np.argwhere(scene.occ)*RES
    static_clear=min([np.min(np.linalg.norm(occ-q,axis=1)) for q in xy]) if len(occ) else 99
    dyn_clear=99
    for k,q in enumerate(xy):
        pred=np.array([[(x+sgn*.5*k*.1)*RES,y*RES] for x,y,sgn in scene.dyn]); dyn_clear=min(dyn_clear,float(np.min(np.linalg.norm(pred-q,axis=1))))
    steer=np.array([u[1] for u in controls]) if controls else np.array([0.])
    curvature=np.tan(steer)/VEH.L
    goal=np.array(scene.goal)*RES; success=int(np.linalg.norm(xy[-1]-goal)<.8 and collisions==0)
    return dict(success=success,path_length=float(seg.sum()),travel_time=len(seg)*VEH.dt,planning_ms=planning_ms,mean_curvature=float(np.mean(np.abs(curvature))),max_curvature=float(np.max(np.abs(curvature))),smoothness=float(np.mean(np.abs(np.diff(curvature)))) if len(curvature)>1 else 0,min_static_clearance=static_clear,min_dynamic_clearance=dyn_clear)

def ci95(x):
    x=np.asarray(x,float); return 1.96*x.std(ddof=1)/math.sqrt(len(x)) if len(x)>1 else 0

def run(n,out):
    os.makedirs(out,exist_ok=True); methods=['A*+DWA','Hybrid A*+TEB','Greedy+APF','RCRA*+DWA','A*+UA','RCRA*+UA (Ours)']; rows=[]
    for i in range(n):
        split='train' if i<n//4 else 'validation' if i<n//2 else 'test'; difficulty=['simple','moderate','hard'][i%3]; s=make_scene(3000+i)
        planning_scene=inflate_scene(s)
        for m in methods:
            g,p,ms=plan(planning_scene,m); states,u,c=track(s,p); r=eval_one(s,states,u,ms,c); r.update(scene=i,split=split,difficulty=difficulty,method=m,footprint_collisions=c); rows.append(r)
    fields=list(rows[0]);
    with open(os.path.join(out,'ackermann_metrics.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    numeric=['success','path_length','travel_time','planning_ms','mean_curvature','max_curvature','smoothness','min_static_clearance','min_dynamic_clearance','footprint_collisions']; summ=[]
    test=[r for r in rows if r['split']=='test']
    for m in methods:
        z=[r for r in test if r['method']==m]; d={'method':m,'test_scenes':len(z)}
        for k in numeric:d[k+'_mean']=np.mean([r[k] for r in z]);d[k+'_ci95']=ci95([r[k] for r in z])
        summ.append(d)
    with open(os.path.join(out,'ackermann_summary.csv'),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
    print(os.path.join(out,'ackermann_summary.csv'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scenes',type=int,default=30);ap.add_argument('--out',default='results');a=ap.parse_args();run(a.scenes,a.out)
