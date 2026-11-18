import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root=DATA; R=6371008.8
def load(part,t):
    b=f"{root}/{part}/{t:08d}/{t:08d}"; rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    return rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
def bearing(v): return np.degrees(np.arctan2(v[:,0],v[:,1]))%360
fig,axs=plt.subplots(1,4,figsize=(22,6))
for ax,(part,t) in zip(axs,[("train",11),("train",81),("train",48),("student_test",56)]):
    tr,nd,ar,rt=load(part,t); lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
    xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
    P=xy(tr); N=xy(nd); A=N[ar[rt,0]]; B=N[ar[rt,1]]; AB=B-A; L2=np.maximum((AB**2).sum(1),1e-12)
    # does the route contain both u->v and v->u of the same road?
    s=set(map(tuple,ar[rt])); both=sum((v,u) in s for u,v in ar[rt])
    # heading-aware: nearest route arc among those within 90° of GPS heading
    h=np.full(len(P),np.nan); v=P[4:]-P[:-4]; sp=np.linalg.norm(v,axis=1)/np.maximum(tr[4:,2]-tr[:-4,2],1); hh=bearing(v); hh[sp<3]=np.nan; h[2:-2]=hh
    AP=P[:,None]-A[None]; tt=np.clip((AP*AB[None]).sum(2)/L2,0,1); D=np.linalg.norm(AP-tt[...,None]*AB[None],axis=2)
    ab=bearing(AB); ok=~np.isnan(h)
    diff=np.abs((h[:,None]-ab[None]+180)%360-180)
    Dh=np.where(diff<90,D,np.inf)
    near_any=D.min(1)<20; near_same=Dh.min(1)<20
    k=np.where(ok,Dh.argmin(1),D.argmin(1)); m=ok&near_same
    kk=k[m]; rho=pd.Series(np.arange(len(kk))).corr(pd.Series(kk),method="spearman")
    print(f"{part} {t}: route arcs {len(rt)}, arcs whose reverse is also in route: {both}; moving pts near route {np.sum(ok&near_any)}, of which with a SAME-direction route arc within 20 m: {np.sum(ok&near_same)} ({np.sum(ok&near_same)/max(1,np.sum(ok&near_any)):.3f}); heading-aware Spearman(time,index)={rho:.3f}; backward jumps>2: {(np.diff(kk)<-2).sum()}")
    ax.plot(*np.array([N[ar[a]] for a in rt]).reshape(-1,2).T,c="g",lw=.5,alpha=.3)
    for a in rt: ax.plot(*N[ar[a]].T,c="g",lw=2)
    sc=ax.scatter(P[:,0],P[:,1],c=tr[:,2],s=2,cmap="viridis"); ax.set_aspect("equal"); ax.set_title(f"{part} {t}")
plt.savefig("investigation_route_dir_check.png",dpi=70)
