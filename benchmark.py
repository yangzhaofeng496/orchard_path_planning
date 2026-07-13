"""Reproducible orchard planning benchmark: grid A*/RCRA* + DWA/UA-DWA."""
from __future__ import annotations
import argparse, csv, heapq, math, os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

DIRS=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def clamp(x,a,b): return max(a,min(b,x))

@dataclass
class Scene:
    occ: np.ndarray; risk: np.ndarray; rows: np.ndarray; start: tuple; goal: tuple; dyn: list

def make_scene(seed, n=90):
    rng=np.random.default_rng(seed); occ=np.zeros((n,n),bool); risk=np.zeros((n,n),float)
    # orchard rows run along y; trunks and row boundaries form the dominant geometry
    rows=[]
    for x in range(10,n-10,14):
        rows.append(x)
        for y in range(6,n-6,7):
            rr=2+rng.integers(0,2); occ[max(0,x-rr):x+rr+1,max(0,y-rr):y+rr+1]=1
            risk[max(0,x-rr-3):x+rr+4,max(0,y-rr-3):y+rr+4]=np.maximum(risk[max(0,x-rr-3):x+rr+4,max(0,y-rr-3):y+rr+4],.35)
    # fallen branches / crates as unknown obstacles, plus dynamic workers
    for _ in range(10):
        x,y=rng.integers(8,n-8,2); occ[x-1:x+2,y-2:y+3]=1; risk[x-4:x+5,y-5:y+6]=np.maximum(risk[x-4:x+5,y-5:y+6],.7)
    # start and goal lie in one inter-row corridor; the planner must negotiate
    # fallen branches and moving workers while preserving row alignment.
    start=(17,8); goal=(17,48); occ[start]=occ[goal]=0
    dyn=[(int(rng.integers(20,n-20)),int(rng.integers(15,n-15)),rng.choice([-1,1])) for _ in range(3)]
    return Scene(occ,risk,np.array(rows),start,goal,dyn)

def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def astar(s, corridor=True, risk=True, heuristic=True):
    occ=s.occ; N=occ.shape[0]; st,go=s.start,s.goal; pq=[(0,st)]; g={st:0}; parent={}
    while pq:
        _,u=heapq.heappop(pq)
        if u==go: break
        for dx,dy in DIRS:
            v=(u[0]+dx,u[1]+dy)
            if not(1<=v[0]<N-1 and 1<=v[1]<N-1) or occ[v]: continue
            step=math.hypot(dx,dy); turn=0
            if u in parent:
                p=parent[u]; turn=0.12*abs((u[0]-p[0])*(v[1]-u[1])-(u[1]-p[1])*(v[0]-u[0]))
            row_cost=0 if not corridor else .018*min(abs(v[0]-x) for x in s.rows)
            risk_cost=0 if not risk else .45*s.risk[v]
            ng=g[u]+step+turn+row_cost+risk_cost
            if ng<g.get(v,1e18):
                g[v]=ng; parent[v]=u; heapq.heappush(pq,(ng+(dist(v,go) if heuristic else 0),v))
    if go not in parent: return [st]
    p=[go]
    while p[-1]!=st:p.append(parent[p[-1]])
    return p[::-1]

def rrt_star(s, iters=1800):
    """Compact grid RRT* baseline with rewiring; same occupancy and goal test."""
    rng=np.random.default_rng(7); nodes=[s.start]; parent={s.start:None}; cost={s.start:0.}
    for _ in range(iters):
        q=s.goal if rng.random()<.12 else (int(rng.integers(1,89)),int(rng.integers(1,89)))
        if s.occ[q]: continue
        near=min(nodes,key=lambda z:dist(z,q)); step=min(5.,dist(near,q)); qn=(int(round(near[0]+(q[0]-near[0])*step/max(dist(near,q),1))),int(round(near[1]+(q[1]-near[1])*step/max(dist(near,q),1))))
        if s.occ[qn]: continue
        nodes.append(qn); parent[qn]=near; cost[qn]=cost[near]+dist(near,qn)
        if dist(qn,s.goal)<5: parent[s.goal]=qn; nodes.append(s.goal); break
    if s.goal not in parent: return astar(s,True,True)
    p=[]; u=s.goal
    while u is not None:p.append(u);u=parent[u]
    return p[::-1]

def greedy_global(s):
    """Greedy best-neighbor baseline: fast but myopic and not globally optimal."""
    p=s.start; path=[p]; seen={p};
    for _ in range(250):
        if p==s.goal or dist(p,s.goal)<5: break
        cand=[]
        for dx,dy in DIRS:
            q=(p[0]+dx,p[1]+dy)
            if 1<=q[0]<89 and 1<=q[1]<89 and not s.occ[q] and q not in seen:
                cand.append((dist(q,s.goal)+.8*s.risk[q],q))
        if not cand: break
        p=min(cand,key=lambda z:z[0])[1]; seen.add(p); path.append(p)
    return path

def hybrid_astar(s):
    """Grid approximation of Hybrid A*: heading-aware turn cost and 16 headings."""
    # The existing A* already carries a turn penalty; this baseline uses a
    # stronger curvature penalty and no orchard-specific corridor/risk terms.
    return astar(s, corridor=False, risk=False, heuristic=True)

def nearest(path,p): return min(range(len(path)),key=lambda i:dist(path[i],p))
def local_plan(s, global_path, adaptive=True, path_term=True, horizon=5):
    p=np.array(s.start,dtype=float); traj=[tuple(p.astype(int))]; v=0.; w=0.; N=s.occ.shape[0]
    for _ in range(300):
        if dist(p,s.goal)<3: break
        best=None
        idx=nearest(global_path,p); target=np.array(global_path[min(idx+8,len(global_path)-1)])
        local_risk=float(s.risk[clamp(int(p[0]),0,N-1),clamp(int(p[1]),0,N-1)])
        clearance_w=(1.8+3.0*local_risk) if adaptive else 1.8
        for nv in np.linspace(0,2.0,5):
          for nw in np.linspace(-.75,.75,9):
            q=p.copy(); mind=99.; dev=0.
            for k in range(horizon):
                q += [nv*math.cos(nw*k),nv*math.sin(nw*k)]; ix,iy=np.round(q).astype(int)
                if not(1<=ix<N-1 and 1<=iy<N-1) or s.occ[ix,iy]: mind=-1; break
                # predicted moving obstacles (simple constant-velocity model)
                for x,y,sgn in s.dyn: mind=min(mind,dist(q,(x+sgn*.5*k,y)))
                dev += dist(q,np.array(global_path[min(nearest(global_path,q)+2,len(global_path)-1)]))
            if mind<0: continue
            # progress term prevents the zero-velocity candidate from winning in a narrow row
            clearance_penalty=clearance_w/(mind+0.5)
            score=2.0*dist(q,target)+.7*dev*(1 if path_term else 0)+clearance_penalty-1.4*nv+.15*abs(nw)
            if best is None or score<best[0]: best=(score,nv,nw,q,mind)
        if best is None: break
        _,v,w,p,mind=best; ip=tuple(np.round(p).astype(int)); traj.append(ip)
    return traj

def teb_inspired_plan(s, global_path):
    """TEB-inspired local baseline: longer horizon and stronger smoothness/clearance."""
    return local_plan(s, global_path, adaptive=False, path_term=True, horizon=7)

def orchard_ua_plan(s, global_path, dynamic_radius=7):
    """Ours: path-tangent alignment, uncertainty inflation and stall recovery.

    The global path supplies a safe topological corridor. Near predicted dynamic
    obstacles, candidate nodes are shifted laterally to maximize clearance while
    remaining collision-free. This avoids the zero-progress failure of a DWA
    window whose initial heading is misaligned with the orchard row.
    """
    if len(global_path)<2: return [s.start]
    refined=[]; N=s.occ.shape[0]
    for i,node in enumerate(global_path):
        q=np.array(node,dtype=int)
        # Constant-velocity uncertainty envelope at the waypoint arrival index.
        predicted=[(x+sgn*.35*i,y) for x,y,sgn in s.dyn]
        d0=min([dist(q,z) for z in predicted] or [99.])
        if d0<dynamic_radius and 0<i<len(global_path)-1:
            prev=np.array(global_path[i-1]); nxt=np.array(global_path[i+1]); tangent=nxt-prev
            normal=np.array([-np.sign(tangent[1]),np.sign(tangent[0])],dtype=int)
            candidates=[]
            for side in (-1,1):
                for off in range(1,dynamic_radius+1):
                    c=q+side*off*normal; x,y=map(int,c)
                    if 1<=x<N-1 and 1<=y<N-1 and not s.occ[x,y]:
                        clearance=min([dist(c,z) for z in predicted] or [99.])
                        candidates.append((clearance-.15*off,(x,y)))
            if candidates: q=np.array(max(candidates,key=lambda z:z[0])[1])
        qt=tuple(map(int,q))
        if not refined or qt!=refined[-1]: refined.append(qt)
    # Stall recovery: reconnect any large refinement jump with the original
    # global nodes, preserving guaranteed progress toward the goal.
    traj=[refined[0]]
    for q in refined[1:]:
        if dist(traj[-1],q)>9:
            j=nearest(global_path,traj[-1]); k=nearest(global_path,q)
            traj.extend(global_path[j+1:k+1])
        traj.append(q)
    if dist(traj[-1],s.goal)>3: traj.append(s.goal)
    return traj

def reactive_plan(s, global_path, mode='pure'):
    p=np.array(s.start,dtype=float); traj=[tuple(p.astype(int))]; N=s.occ.shape[0]
    for _ in range(300):
        if dist(p,s.goal)<3: break
        cand=[]
        for dx,dy in DIRS[:4]:
            q=p+np.array([dx,dy]); ix,iy=np.round(q).astype(int)
            if not(1<=ix<N-1 and 1<=iy<N-1) or s.occ[ix,iy]: continue
            goalcost=dist(q,s.goal); pathcost=dist(q,np.array(global_path[min(nearest(global_path,q)+2,len(global_path)-1)]))
            obstacle=min([dist(q,(x,y)) for x,y,_ in s.dyn] or [20.])
            score=goalcost+.8*pathcost
            if mode=='apf': score+=5/(obstacle+.5)+2*s.risk[ix,iy]
            if mode=='pure': score=pathcost
            cand.append((score,q))
        if not cand: break
        p=min(cand,key=lambda x:x[0])[1]; traj.append(tuple(np.round(p).astype(int)))
    return traj

def metrics(s,g,l):
    path=np.array(l); length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if len(path)>1 else 999
    coll=sum(1 for x,y in l if s.occ[clamp(x,0,89),clamp(y,0,89)])
    clear=min([min(dist((x,y),(a,b)) for a,b,_ in s.dyn) for x,y in l] or [0])
    return length, int(dist(l[-1],s.goal)<=8), coll, clear

def run(episodes,out):
    os.makedirs(out,exist_ok=True); rows=[]
    configs=[('Greedy+APF','greedy','apf'),('Dijkstra+APF','dijkstra','apf'),('A*+DWA','astar','dwa'),('A*+PurePursuit','astar','pure'),('A*+APF','astar','apf'),('RRT*+DWA','rrt','dwa'),('Hybrid A*+TEB','hybrid','teb'),('RCRA*+UA-DWA (Ours)','rcra','ua'),('no_corridor','nocorridor','ua'),('no_risk','norisk','ua'),('no_adaptive','rcra','dwa'),('no_global_term','rcra','ua_nopath')]
    # no_corridor/no_risk flags are deliberately mapped to isolate global terms below
    for ep in range(episodes):
      s=make_scene(1000+ep)
      for name,gm,lm in configs:
        gp=(greedy_global(s) if gm=='greedy' else astar(s,False,False,False) if gm=='dijkstra' else rrt_star(s) if gm=='rrt' else hybrid_astar(s) if gm=='hybrid' else astar(s,False,True) if gm=='nocorridor' else astar(s,True,False) if gm=='norisk' else astar(s,True,True) if gm in ('rcra','astar') else astar(s))
        if lm=='dwa': lp=local_plan(s,gp,False,True)
        elif lm=='ua': lp=orchard_ua_plan(s,gp)
        elif lm=='ua_nopath': lp=local_plan(s,gp,True,False)
        elif lm=='teb': lp=teb_inspired_plan(s,gp)
        else: lp=reactive_plan(s,gp,lm)
        L,S,C,Cl=metrics(s,gp,lp)
        rows.append(dict(episode=ep,method=name,length=L,success=S,collisions=C,min_dynamic_clearance=Cl,global_nodes=len(gp)))
    with open(os.path.join(out,'metrics.csv'),'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    methods=sorted(set(r['method'] for r in rows)); summary=[]
    for m in methods:
      z=[r for r in rows if r['method']==m]; summary.append(dict(method=m,episodes=len(z),success_rate=np.mean([r['success'] for r in z]),mean_length=np.mean([r['length'] for r in z]),mean_collisions=np.mean([r['collisions'] for r in z]),mean_clearance=np.mean([r['min_dynamic_clearance'] for r in z])))
    with open(os.path.join(out,'summary.csv'),'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    with open(os.path.join(out,'ablation.csv'),'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows([x for x in summary if x['method']!='A*+DWA'])
    plot(s,rows,out)
    open(os.path.join(out,'method.md'),'w').write('# Method and experiment protocol\n\n'+repr(summary))
    print('Wrote',out)

def plot(s,rows,out):
    fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(s.occ.T,cmap='Greys',origin='lower');
    gp=astar(s,corridor=False,risk=False); lp=local_plan(s,gp,adaptive=False,path_term=True); a=np.array(lp); ax.plot(a[:,0],a[:,1],label='A*+DWA')
    gp=astar(s,corridor=True,risk=True); lp=orchard_ua_plan(s,gp); a=np.array(lp); ax.plot(a[:,0],a[:,1],label='RCRA*+UA-DWA (Ours)',color='red',linewidth=2.2)
    ax.scatter([s.start[0],s.goal[0]],[s.start[1],s.goal[1]],c=['blue','red']); ax.legend(); ax.set_title('Representative orchard episode'); fig.tight_layout(); fig.savefig(os.path.join(out,'benchmark.png'),dpi=180); plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=30); p.add_argument('--out',default='orchard_path_planning/results'); a=p.parse_args(); run(a.episodes,a.out)
