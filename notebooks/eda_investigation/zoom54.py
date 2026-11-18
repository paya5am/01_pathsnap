import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
root=os.path.join(DATA, "train"); R=6371008.8
def load(t):
    b=f"{root}/{t:08d}/{t:08d}"; rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    return rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
tr,nd,ar,rt=load(54); lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
P=xy(tr); N=xy(nd)
fig,axs=plt.subplots(1,3,figsize=(18,6))
for ax,i in zip(axs,[3000,9000,15000]):
    cx,cy=P[i]; w=500
    m=(abs(N[:,0]-cx)<w)&(abs(N[:,1]-cy)<w)
    sel=m[ar[:,0]]|m[ar[:,1]]
    for u,v in ar[sel]: ax.plot(*N[[u,v]].T,c="0.7",lw=.6)
    for a in rt:
        u,v=ar[a]
        if m[u] or m[v]: ax.plot(*N[[u,v]].T,c="g",lw=2.5)
    pm=(abs(P[:,0]-cx)<w)&(abs(P[:,1]-cy)<w); ax.plot(P[pm,0],P[pm,1],".",c="b",ms=2)
    ax.set_xlim(cx-w,cx+w); ax.set_ylim(cy-w,cy+w); ax.set_aspect("equal"); ax.set_title(f"trip 54 around point {i}")
plt.savefig("investigation_zoom54.png",dpi=80)
# estimate a constant offset: median vector from GPS point to nearest route point
A=N[ar[rt,0]];B=N[ar[rt,1]];AB=B-A;L2=np.maximum((AB**2).sum(1),1e-12)
sub=P[::20]; AP=sub[:,None]-A[None]; t=np.clip((AP*AB[None]).sum(2)/L2,0,1); Q=A[None]+t[...,None]*AB[None]
d=np.linalg.norm(sub[:,None]-Q,axis=2); k=d.argmin(1); vec=Q[np.arange(len(sub)),k]-sub
print("median offset vector (m):",np.median(vec,0), " median |d|:",np.median(d.min(1)))
s=sub+np.median(vec,0); AP=s[:,None]-A[None]; t=np.clip((AP*AB[None]).sum(2)/L2,0,1); d2=np.linalg.norm(AP-t[...,None]*AB[None],axis=2).min(1)
print("after shifting GPS by that vector, median d2route:",np.median(d2), "p90",np.percentile(d2,90))
print("trip 54 lon/lat center",lon0,lat0)
