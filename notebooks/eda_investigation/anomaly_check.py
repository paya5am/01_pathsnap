import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
root=os.path.join(DATA, "train")
R=6371008.8
def load(t):
    b=f"{root}/{t:08d}/{t:08d}"
    rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    return rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
fig,axs=plt.subplots(2,3,figsize=(18,11))
for ax,t in zip(axs.ravel(),[54,33,2,21,32,13]):
    tr,nd,ar,rt=load(t); lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
    xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
    P=xy(tr); N=xy(nd); tt=cKDTree(P)
    mid=(N[ar[:,0]]+N[ar[:,1]])/2; near=tt.query(mid)[0]<400
    for u,v in ar[near]: ax.plot(*N[[u,v]].T,c="0.8",lw=.5)
    for a in rt: u,v=ar[a]; ax.plot(*N[[u,v]].T,c="g",lw=2)
    sc=ax.scatter(P[:,0],P[:,1],c=tr[:,2],s=1,cmap="viridis")
    dt=np.diff(tr[:,2]); sp=np.linalg.norm(np.diff(P,axis=0),axis=1)/dt
    ax.set_title(f"trip {t}: route {len(rt)} arcs ({len(np.unique(rt))} uniq); speed p99 {np.percentile(sp,99):.0f} max {sp.max():.0f} m/s",fontsize=9)
    ax.set_aspect("equal")
plt.colorbar(sc,ax=axs.ravel().tolist(),label="time (s)",shrink=.5)
plt.savefig("investigation_inspect_anomalies.png",dpi=90)
# trip 21: does GPS revisit route arcs? sequence of nearest route-arc index
for t in [21,84,2,33]:
    tr,nd,ar,rt=load(t); lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
    xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
    P=xy(tr); N=xy(nd); A=N[ar[rt,0]]; B=N[ar[rt,1]]
    AB=B-A; L2=np.maximum((AB**2).sum(1),1e-12)
    idx=[]
    for s in range(0,len(P),2000):
        AP=P[s:s+2000,None]-A[None]; tt_=np.clip((AP*AB[None]).sum(2)/L2,0,1); d=np.linalg.norm(AP-tt_[...,None]*AB[None],axis=2); idx.append(d.argmin(1))
    idx=np.concatenate(idx)
    q=np.linspace(0,len(P)-1,25).astype(int)
    print(t,"route idx of nearest route arc at 25 evenly spaced times:",idx[q].tolist())
    print("   GPS time span covered by route-start/end region: first pt idx",idx[0],"last",idx[-1],"of",len(rt))
