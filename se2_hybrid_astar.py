"""Risk-corridor SE(2) Hybrid A* for the NWD01 Ackermann platform."""
from __future__ import annotations
import heapq,math
from dataclasses import dataclass
import numpy as np
from ackermann_model import AckermannModel

@dataclass
class CorridorState:
    x: float; y: float; yaw: float; risk: float; clearance: float

class SE2HybridAStar:
    def __init__(self,model=None,resolution=.4,yaw_bins=24,primitive_steps=6):
        self.model=model or AckermannModel(dt=.1);self.res=resolution;self.yaw_bins=yaw_bins;self.steps=primitive_steps
    def key(self,s):
        return (int(round(s[0]/self.res)),int(round(s[1]/self.res)),int(round(((s[2]+math.pi)%(2*math.pi))/(2*math.pi)*self.yaw_bins))%self.yaw_bins)
    def footprint_free(self,scene,state):
        c=self.model.get_corners(state)/self.res; pts=[np.asarray(state[:2])/self.res]
        for a,b in zip(c,np.roll(c,-1,axis=0)):
            for u in np.linspace(0,1,9):pts.append(a*(1-u)+b*u)
        for x,y in pts:
            ix,iy=int(round(x)),int(round(y))
            if ix<0 or iy<0 or ix>=scene.occ.shape[0] or iy>=scene.occ.shape[1] or scene.occ[ix,iy]:return False
        return True
    def primitive(self,scene,state,v,delta):
        tr=self.model.simulate(state,v,delta,self.steps)
        return tr if all(self.footprint_free(scene,q) for q in tr[1:]) else None
    def clearance(self,scene,state,max_r=12):
        x,y=np.asarray(state[:2])/self.res; best=max_r
        for ix in range(max(0,int(x)-max_r),min(scene.occ.shape[0],int(x)+max_r+1)):
            for iy in range(max(0,int(y)-max_r),min(scene.occ.shape[1],int(y)+max_r+1)):
                if scene.occ[ix,iy]:best=min(best,math.hypot(ix-x,iy-y))
        return best*self.res
    def plan(self,scene,start=None,goal=None,max_expansions=40000):
        start=np.array(start if start is not None else [scene.start[0]*self.res,scene.start[1]*self.res,math.pi/2],float)
        goal=np.array(goal if goal is not None else [scene.goal[0]*self.res,scene.goal[1]*self.res,math.pi/2],float)
        sk=self.key(start);pq=[(0.,0,sk)];states={sk:start};g={sk:0.};parent={};control={};counter=0;goal_key=None
        for _ in range(max_expansions):
            if not pq:break
            _,_,k=heapq.heappop(pq);s=states[k]
            if np.linalg.norm(s[:2]-goal[:2])<.7 and abs(self.model.normalize_angle(s[2]-goal[2]))<.7:goal_key=k;break
            for v,delta in self.model.get_motion_primitives():
                tr=self.primitive(scene,s,v,delta)
                if tr is None:continue
                ns=tr[-1];nk=self.key(ns);ix,iy=int(round(ns[0]/self.res)),int(round(ns[1]/self.res))
                risk=float(scene.risk[np.clip(ix,0,89),np.clip(iy,0,89)]);row_heading=abs(math.cos(ns[2]))
                reverse=1.4 if v<0 else 0.;turn=.45*abs(delta)/self.model.max_steer;clear=self.clearance(scene,ns);safety=.7/(clear+.1)
                ng=g[k]+abs(v)*self.model.dt*self.steps+turn+reverse+.7*risk+.35*row_heading+safety
                if ng<g.get(nk,1e18):
                    g[nk]=ng;states[nk]=ns;parent[nk]=k;control[nk]=(v,delta);counter+=1
                    h=np.linalg.norm(ns[:2]-goal[:2])+1.2*abs(self.model.normalize_angle(ns[2]-goal[2]))
                    heapq.heappush(pq,(ng+h,counter,nk))
        if goal_key is None:return [],{'success':False,'expanded':len(g)}
        keys=[];k=goal_key
        while k!=sk:keys.append(k);k=parent[k]
        keys.append(sk);keys.reverse();path=[states[k] for k in keys]
        corridor=[]
        for s in path:
            ix,iy=int(round(s[0]/self.res)),int(round(s[1]/self.res));risk=float(scene.risk[np.clip(ix,0,89),np.clip(iy,0,89)])
            corridor.append(CorridorState(float(s[0]),float(s[1]),float(s[2]),risk,self.clearance(scene,s)))
        return corridor,{'success':True,'expanded':len(g),'cost':g[goal_key]}

