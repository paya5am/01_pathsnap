"""Candidate-radius analysis. Selection uses TRAIN trips only; student_test is only used to verify."""
import os, sys, math, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
R = 6371008.8; STEP = 5.0; RMAX = 150.0
def bearing(v): return np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
def seg_dist(P, A, B):
    AB = B - A; AP = P - A
    t = np.clip((AP * AB).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(AP - t[..., None] * AB, axis=-1)

rows = []
for part in ["train", "student_test"]:
    for tid in sorted(os.listdir(f"{ROOT}/{part}")):
        b = f"{ROOT}/{part}/{tid}/{tid}"; rd = lambda e: pd.read_csv(f"{b}.{e}", sep=r"\s+", header=None).values
        tr, nd, ar, rt = rd("track"), rd("nodes"), rd("arcs").astype(int), rd("route").ravel().astype(int)
        lon0, lat0 = tr[:, 0].mean(), tr[:, 1].mean(); c = math.cos(math.radians(lat0))
        xy = lambda a: np.column_stack([np.radians(a[:, 0] - lon0) * R * c, np.radians(a[:, 1] - lat0) * R])
        P, N = xy(tr), xy(nd)
        # candidate arcs: drop zero-length, merge exact duplicate (u,v) pairs
        L = np.linalg.norm(N[ar[:, 1]] - N[ar[:, 0]], axis=1)
        keep = np.where(L >= 0.01)[0]
        _, first = np.unique(ar[keep], axis=0, return_index=True); keep = keep[np.sort(first)]
        # GPS heading (4-sample baseline) where moving > 3 m/s, else unknown
        h = np.full(len(P), np.nan)
        if len(P) > 4:
            v = P[4:] - P[:-4]; sp = np.linalg.norm(v, axis=1) / np.maximum(tr[4:, 2] - tr[:-4, 2], 1)
            hh = bearing(v); hh[sp < 3] = np.nan; h[2:-2] = hh
        # d* = distance to the reference arc: nearest SAME-direction route arc if heading known, else nearest route arc
        A, B = N[ar[rt, 0]], N[ar[rt, 1]]; ab = bearing(B - A)
        dstar = np.empty(len(P))
        for s in range(0, len(P), 1500):
            sl = slice(s, s + 1500)
            D = seg_dist(P[sl, None], A[None], B[None])
            diff = np.abs((h[sl, None] - ab[None] + 180) % 360 - 180)
            Ds = np.where(np.isnan(h[sl, None]) | (diff < 90), D, np.inf)
            dstar[sl] = Ds.min(1)
        # all candidate arcs within RMAX: densify, ball query, exact distance
        lo, hi = P.min(0) - RMAX - 50, P.max(0) + RMAX + 50
        Ak, Bk = N[ar[keep, 0]], N[ar[keep, 1]]
        inbox = np.where((np.maximum(Ak, Bk) >= lo).all(1) & (np.minimum(Ak, Bk) <= hi).all(1))[0]
        Lk = np.linalg.norm(Bk[inbox] - Ak[inbox], axis=1); k = np.maximum(1, np.ceil(Lk / STEP).astype(int))
        rep = np.repeat(np.arange(len(inbox)), k + 1); t = np.concatenate([np.linspace(0, 1, kk + 1) for kk in k])
        dp = Ak[inbox][rep] + t[:, None] * (Bk[inbox] - Ak[inbox])[rep]; darc = inbox[rep]
        tree = cKDTree(dp)
        K = {r: np.zeros(len(P), int) for r in range(5, 151, 5)}
        dnear = np.full(len(P), np.inf); dsecond_pool = []
        Kadapt = {}
        for s in range(0, len(P), 2000):
            hits = tree.query_ball_point(P[s:s + 2000], r=RMAX + STEP / 2)
            pi = np.repeat(np.arange(len(hits)), [len(x) for x in hits]) + s
            ai = darc[np.concatenate([np.asarray(x, int) for x in hits])] if len(pi) else np.array([], int)
            key = np.unique(pi.astype(np.int64) * 10_000_000 + ai)
            pi, ai = key // 10_000_000, key % 10_000_000
            d = seg_dist(P[pi], Ak[ai], Bk[ai])
            m = d <= RMAX; pi, d = pi[m], d[m]
            for r in K: K[r][s:s + 2000] = np.bincount(pi[d <= r] - s, minlength=min(2000, len(P) - s))
            mn = np.full(min(2000, len(P) - s), np.inf); np.minimum.at(mn, pi - s, d); dnear[s:s + 2000] = mn
            for delta in (5, 10, 15, 20, 25, 30, 40, 50):
                thr = np.minimum(mn[pi - s] + delta, RMAX)
                Kadapt.setdefault(delta, np.zeros(len(P), int))[s:s + 2000] = np.bincount(pi[d <= thr] - s, minlength=len(mn))
        df = pd.DataFrame({"partition": part, "trip_id": int(tid), "dstar": dstar, "dnear": dnear,
                           "moving": ~np.isnan(h), **{f"K{r}": K[r] for r in K}, **{f"KA{dl}": Kadapt[dl] for dl in Kadapt}})
        rows.append(df)
        print(part, tid, len(P), flush=True)
pd.concat(rows, ignore_index=True).to_parquet(f"{OUT}/radius_points.parquet") if False else pd.concat(rows, ignore_index=True).to_csv(f"{OUT}/radius_points.csv.gz", index=False)
