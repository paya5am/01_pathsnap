import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
R=6371008.8
def bearing(v): return np.degrees(np.arctan2(v[:,0],v[:,1]))%360
fig,axs=plt.subplots(1,2,figsize=(16,8))
for ax,(part,t) in zip(axs,[("train",66),("student_test",9)]):
    b=f"{DATA}/{part}/{t:08d}/{t:08d}"; rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    tr,nd,ar,rt=rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
    lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
    xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
    P=xy(tr); N=xy(nd); A=N[ar[rt,0]]; B=N[ar[rt,1]]; AB=B-A; L2=np.maximum((AB**2).sum(1),1e-12); ab=bearing(AB)
    h=np.full(len(P),np.nan); v=P[4:]-P[:-4]; sp=np.full(len(P),np.nan); s_=np.linalg.norm(v,axis=1)/np.maximum(tr[4:,2]-tr[:-4,2],1); hh=bearing(v); hh[s_<3]=np.nan; h[2:-2]=hh; sp[2:-2]=s_
    AP=P[:,None]-A[None]; tt=np.clip((AP*AB[None]).sum(2)/L2,0,1); D=np.linalg.norm(AP-tt[...,None]*AB[None],axis=2)
    diff=np.abs((h[:,None]-ab[None]+180)%360-180); Ds=np.where(diff<90,D,np.inf)
    near=~np.isnan(h)&(D.min(1)<20); bad=near&(Ds.min(1)>=20)
    idx=np.where(bad)[0]; runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1); r=max(runs,key=len)
    k=D.argmin(1)
    print(f"{part} {t}: longest run {len(r)} pts, idx {r[0]}-{r[-1]}, t {tr[r[0],2]:.0f}-{tr[r[-1],2]:.0f}s, speed med {np.nanmedian(sp[r]):.1f} m/s, dist to nearest route arc med {np.median(D.min(1)[r]):.1f} m")
    print("   nearest route arcs (route positions):",np.unique(k[r]).tolist()[:12], " heading diff to them median",np.median(diff[r,k[r]]).round(0))
    # is there ANY network arc in GPS direction nearby?
    Aall=N[ar[:,0]];Ball=N[ar[:,1]];ABa=Ball-Aall;L2a=np.maximum((ABa**2).sum(1),1e-12);aba=bearing(ABa)
    mid=P[r]; sel=np.where(np.linalg.norm((Aall+Ball)/2-mid.mean(0),axis=1)<500)[0]
    APa=mid[:,None]-Aall[sel][None]; tta=np.clip((APa*ABa[sel][None]).sum(2)/L2a[sel],0,1); Da=np.linalg.norm(APa-tta[...,None]*ABa[sel][None],axis=2)
    da=np.abs((h[r][:,None]-aba[sel][None]+180)%360-180); Dsa=np.where(da<90,Da,np.inf)
    print("   any network arc in GPS direction within 20 m for",np.mean(Dsa.min(1)<20).round(3),"of run points; nearest such arc median dist",np.median(Dsa.min(1)).round(1),"m")
    cx,cy=P[r].mean(0); w=250
    m=(np.abs((N[ar[:,0]]+N[ar[:,1]])/2-[cx,cy])<w).all(1)
    for u,v_ in ar[m]: ax.annotate("",xy=N[v_],xytext=N[u],arrowprops=dict(arrowstyle="->",color="0.6",lw=.7))
    for a in rt:
        u,v_=ar[a]
        if (np.abs((N[u]+N[v_])/2-[cx,cy])<w).all(): ax.annotate("",xy=N[v_],xytext=N[u],arrowprops=dict(arrowstyle="->",color="g",lw=2))
    q=(np.abs(P-[cx,cy])<w).all(1); ax.scatter(P[q,0],P[q,1],c=tr[q,2],s=8,cmap="viridis"); ax.scatter(P[r,0],P[r,1],facecolors="none",edgecolors="r",s=40)
    ax.set_xlim(cx-w,cx+w); ax.set_ylim(cy-w,cy+w); ax.set_aspect("equal"); ax.set_title(f"{part} {t}: arcs (grey arrows), route (green arrows), GPS by time, run in red")
plt.savefig("investigation_run_check.png",dpi=70)
