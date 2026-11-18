import os
# Folder containing train/ and student_test/ (env var PATHSNAP_DATA)
DATA = os.environ.get("PATHSNAP_DATA", "Student folder")
import numpy as np, pandas as pd, math
root=DATA; R=6371008.8
def load(part,t):
    b=f"{root}/{part}/{t:08d}/{t:08d}"; rd=lambda e: pd.read_csv(f"{b}.{e}",sep=r"\s+",header=None).values
    return rd("track"),rd("nodes"),rd("arcs").astype(int),rd("route").ravel().astype(int)
import os
# 1) non-integer timestamps
for part in ["train","student_test"]:
    for tid in sorted(os.listdir(f"{root}/{part}")):
        t=pd.read_csv(f"{root}/{part}/{tid}/{tid}.track",sep=r"\s+",header=None)[2].values
        frac=t-np.round(t)
        if np.any(frac!=0):
            nz=frac[frac!=0]
            dt=np.diff(t)
            print(part,tid,"non-integer ts:",len(nz),"of",len(t),"examples",np.unique(np.round(t[frac!=0][:5],4)),"fractional parts",np.unique(np.round(nz,4))[:6], "dt unique sample",np.unique(np.round(dt,3))[:8])
# raw text of one non-integer line
