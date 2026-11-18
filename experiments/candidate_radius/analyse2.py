import numpy as np, pandas as pd
d = pd.read_csv("out/radius_points.csv.gz")
ANOM = [54, 33, 2]   # flagged in M2 EDA independently of any radius (parked off-network / route ends early)
tr = d[d.partition == "train"]; trc = tr[~tr.trip_id.isin(ANOM)]; st = d[d.partition == "student_test"]
def fixed(g, r): return (g.dstar <= r).mean()
def adapt(g, dl, cap=150): return (g.dstar <= np.minimum(g.dnear + dl, cap)).mean()
print("share of train points in trips 54/33/2:", round(tr.trip_id.isin(ANOM).mean(), 3))
print("\nFIXED radius   | train-all | train-clean | trip-weighted median (clean) | student-test | mean K clean | empty sets clean")
for r in (10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100):
    tw = trc.groupby("trip_id").apply(lambda g: fixed(g, r)).median()
    print(f"  r={r:3d} m     | {fixed(tr,r):.4f}    | {fixed(trc,r):.4f}      | {tw:.4f}                       | {fixed(st,r):.4f}       | {trc[f'K{r}'].mean():5.2f}        | {(trc[f'K{r}']==0).mean():.4f}")
print("\nADAPTIVE d_near+delta | train-all | train-clean | student-test | mean K clean | mean K^2 clean | empty sets clean")
for dl in (5, 10, 15, 20, 25, 30, 40, 50):
    print(f"  delta={dl:2d} m        | {adapt(tr,dl):.4f}    | {adapt(trc,dl):.4f}      | {adapt(st,dl):.4f}       | {trc[f'KA{dl}'].mean():5.2f}        | {(trc[f'KA{dl}']**2).mean():7.1f}        | {(trc[f'KA{dl}']==0).mean():.4f}")
# equal-budget comparison on clean train: for each adaptive delta find fixed radius with closest mean K
print("\nEQUAL CANDIDATE BUDGET (train-clean):")
for dl in (10, 15, 20, 25, 30):
    kA = trc[f"KA{dl}"].mean()
    r = min(range(5, 151, 5), key=lambda r: abs(trc[f"K{r}"].mean() - kA))
    print(f"  adaptive delta={dl} (K={kA:.2f}) coverage {adapt(trc,dl):.4f}  vs  fixed r={r} (K={trc[f'K{r}'].mean():.2f}) coverage {fixed(trc,r):.4f}")
# where adaptive misses (clean, delta=20): reference arc further than nearest+20 — how far is it, and are these points moving?
m = trc[trc.dstar > np.minimum(trc.dnear + 20, 150)]
print(f"\nclean misses at delta=20: {len(m)} pts; dstar-dnear median {np.median(m.dstar-m.dnear):.1f} m; moving share {m.moving.mean():.2f}; dstar median {m.dstar.median():.1f} m; trips {m.trip_id.value_counts().head(5).to_dict()}")
