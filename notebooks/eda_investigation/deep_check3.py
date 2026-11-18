import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math
root=os.path.join(DATA, "train"); R=6371008.8
def bearing(v): return np.degrees(np.arctan2(v[:,0],v[:,1]))%360
for t in [82,2,10,3]:
    b=f"{root}/{t:08d}/{t:08d}"; rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    tr,nd,ar,rt=rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
    lon0,lat0=tr[:,0].mean(),tr[:,1].mean(); c=math.cos(math.radians(lat0))
    xy=lambda a: np.column_stack([np.radians(a[:,0]-lon0)*R*c,np.radians(a[:,1]-lat0)*R])
    P=xy(tr); N=xy(nd); A=N[ar[rt,0]]; B=N[ar[rt,1]]; AB=B-A; L2=np.maximum((AB**2).sum(1),1e-12); ab=bearing(AB)
    h=np.full(len(P),np.nan); v=P[4:]-P[:-4]; sp=np.linalg.norm(v,axis=1)/np.maximum(tr[4:,2]-tr[:-4,2],1); hh=bearing(v); hh[sp<3]=np.nan; h[2:-2]=hh
    # turn rate: heading change between (i-4->i) and (i->i+4)
    turn=np.full(len(P),np.nan); v1=P[4:-4]-P[:-8]; v2=P[8:]-P[4:-4]; turn[4:-4]=np.abs((bearing(v2)-bearing(v1)+180)%360-180)
    AP=P[:,None]-A[None]; tt=np.clip((AP*AB[None]).sum(2)/L2,0,1); D=np.linalg.norm(AP-tt[...,None]*AB[None],axis=2)
    diff=np.abs((h[:,None]-ab[None]+180)%360-180); Ds=np.where(diff<90,D,np.inf)
    near=~np.isnan(h)&(D.min(1)<20); bad=near&(Ds.min(1)>=20)
    # runs of consecutive bad points
    idx=np.where(bad)[0]; runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else []
    rl=[len(r) for r in runs]
    # nearest route arc length for bad points
    k=D.argmin(1)
    L=np.sqrt(L2)
    print(f"trip {t}: bad {bad.sum()}/{near.sum()}; median turn at bad pts {np.nanmedian(turn[bad]):.0f}° vs all {np.nanmedian(turn[near]):.0f}°; runs: n={len(rl)}, max len {max(rl) if rl else 0}, runs>=10: {sum(r>=10 for r in rl)}; median speed bad {np.nanmedian(sp[np.clip(idx-2,0,len(sp)-1)]):.1f} m/s; median nearest-route-arc len at bad pts {np.median(L[k[bad]]):.0f} m (all {np.median(L[k[near]]):.0f} m); heading diff to nearest route arc median {np.median(np.abs((h[bad]-ab[k[bad]]+180)%360-180)):.0f}°")
    if rl and max(rl)>=10:
        r=runs[int(np.argmax(rl))]; print("    longest run points",r[0],"-",r[-1],"t=",tr[r[0],2],"-",tr[r[-1],2])
