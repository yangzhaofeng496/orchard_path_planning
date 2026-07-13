"""Risk-corridor guided MPPI controller with kinodynamic recovery."""
from __future__ import annotations
import math
import numpy as np
from ackermann_model import AckermannModel

class RiskCorridorMPPI:
    def __init__(self,model=None,resolution=.4,samples=64,horizon=14,seed=42):
        self.model=model or AckermannModel(dt=.1);self.res=resolution;self.K=samples;self.H=horizon;self.rng=np.random.default_rng(seed)
    def collision(self,scene,state):
        c=self.model.get_corners(state)/self.res
        for a,b in zip(c,np.roll(c,-1,axis=0)):
            for u in np.linspace(0,1,7):
                x,y=a*(1-u)+b*u;ix,iy=int(round(x)),int(round(y))
                if ix<0 or iy<0 or ix>=90 or iy>=90 or scene.occ[ix,iy]:return True
        return False
    def nearest(self,corridor,state):return min(range(len(corridor)),key=lambda i:(corridor[i].x-state[0])**2+(corridor[i].y-state[1])**2)
    def rollout_cost(self,scene,state,controls,corridor,t0,use_dynamic=True):
        s=state.copy();cost=0.;last_delta=0.
        for h,(v,d) in enumerate(controls):
            s=self.model.update(s,v,d);i=min(self.nearest(corridor,s)+2,len(corridor)-1);c=corridor[i]
            path=(s[0]-c.x)**2+(s[1]-c.y)**2;heading=abs(self.model.normalize_angle(s[2]-c.yaw));risk=c.risk
            dyn=0.
            for x,y,sgn in (scene.dyn if use_dynamic else []):
                ox=(x+sgn*.5*(t0+h)*self.model.dt)*self.res;oy=y*self.res;dd=math.hypot(s[0]-ox,s[1]-oy);dyn+=10/(dd+.2)
            if self.collision(scene,s):cost+=5000
            progress=i/max(len(corridor)-1,1)
            cost+=(2+4*risk)*path+(1+2*risk)*heading+dyn+.15*d*d+.3*(d-last_delta)**2-.8*v-2.5*progress;last_delta=d
        goal=corridor[-1];cost+=4*math.hypot(s[0]-goal.x,s[1]-goal.y);return cost
    def recovery(self,scene,state,corridor):
        """Return only a corridor state reachable by a collision-free primitive."""
        i=self.nearest(corridor,state)
        for target in corridor[i+1:min(len(corridor),i+10)]:
            desired=math.atan2(target.y-state[1],target.x-state[0]);err=self.model.normalize_angle(desired-state[2]);d=np.clip(err,-self.model.max_steer,self.model.max_steer)
            tr=self.model.simulate(state,.5,d,8)
            if all(not self.collision(scene,q) for q in tr):return tr[-1]
        return None
    def track(self,scene,corridor,max_steps=1400,enable_recovery=True,use_dynamic=True):
        if not corridor:return np.empty((0,3)),{'success':False,'collisions':0,'recoveries':0}
        state=np.array([corridor[0].x,corridor[0].y,corridor[0].yaw]);traj=[state.copy()];u=np.tile([1.0,0.],(self.H,1));stall=0;recoveries=0;collisions=0;prev_goal=1e9
        for t in range(max_steps):
            noise=np.zeros((self.K,self.H,2));noise[:,:,0]=self.rng.normal(0,.25,(self.K,self.H));noise[:,:,1]=self.rng.normal(0,.12,(self.K,self.H))
            candidates=u[None,:,:]+noise;candidates[:,:,0]=np.clip(candidates[:,:,0],-.4,self.model.max_speed);candidates[:,:,1]=np.clip(candidates[:,:,1],-self.model.max_steer,self.model.max_steer)
            costs=np.array([self.rollout_cost(scene,state,c,corridor,t,use_dynamic) for c in candidates]);w=np.exp(-(costs-costs.min())/max(costs.std(),1e-6));w/=w.sum();u=np.sum(w[:,None,None]*candidates,axis=0)
            state=self.model.update(state,*u[0]);u[:-1]=u[1:];u[-1]=u[-2];collisions+=int(self.collision(scene,state));traj.append(state.copy())
            gd=math.hypot(state[0]-corridor[-1].x,state[1]-corridor[-1].y)
            stall=stall+1 if gd>prev_goal-.003 else 0;prev_goal=gd
            if gd<.7:return np.asarray(traj),{'success':collisions==0,'collisions':collisions,'recoveries':recoveries}
            if stall>35 and enable_recovery:
                rec=self.recovery(scene,state,corridor)
                if rec is not None:state=rec;traj.append(state.copy());recoveries+=1
                stall=0
        return np.asarray(traj),{'success':False,'collisions':collisions,'recoveries':recoveries}
