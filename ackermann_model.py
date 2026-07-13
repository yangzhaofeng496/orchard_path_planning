"""Ackermann bicycle model for the Xiaomayi NWD01 platform."""
import numpy as np

class AckermannModel:
    def __init__(self,wheel_base=0.9,vehicle_length=1.62,vehicle_width=0.82,max_steer=0.35,max_speed=1.4,dt=0.1):
        self.L=wheel_base; self.length=vehicle_length; self.width=vehicle_width
        self.max_steer=max_steer; self.max_speed=max_speed; self.dt=dt
    def update(self,state,v,delta):
        x,y,yaw=state; v=np.clip(v,-self.max_speed,self.max_speed); delta=np.clip(delta,-self.max_steer,self.max_steer)
        x+=v*np.cos(yaw)*self.dt; y+=v*np.sin(yaw)*self.dt; yaw+=v/self.L*np.tan(delta)*self.dt
        return np.array([x,y,self.normalize_angle(yaw)])
    def simulate(self,state,v,delta,steps=10):
        out=[np.array(state)]; cur=np.array(state)
        for _ in range(steps): cur=self.update(cur,v,delta); out.append(cur.copy())
        return np.array(out)
    def get_motion_primitives(self):
        steering=[-self.max_steer,-self.max_steer*.5,0,self.max_steer*.5,self.max_steer]
        return [(v,d) for v in [0.8,-0.4] for d in steering]
    def get_corners(self,state):
        x,y,yaw=state; local=np.array([[self.length/2,self.width/2],[self.length/2,-self.width/2],[-self.length/2,-self.width/2],[-self.length/2,self.width/2]])
        rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        c=local@rot.T; c[:,0]+=x; c[:,1]+=y; return c
    @staticmethod
    def normalize_angle(a): return (a+np.pi)%(2*np.pi)-np.pi
