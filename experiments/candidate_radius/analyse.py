import numpy as np, pandas as pd
d = pd.read_csv("out/radius_points.csv.gz")
tr = d[d.partition == "train"]; st = d[d.partition == "student_test"]
pd.set_option("display.width", 220)
# robust GPS noise estimate (Newson & Krumm 2009: sigma = 1.4826 * median distance) on moving train points
mv = tr[tr.moving & np.isfinite(tr.dstar)]
sigma = 1.4826 * np.median(mv.dstar)
print(f"robust sigma (train, moving) = {sigma:.2f} m; empirical quantiles of d*: p90 {mv.dstar.quantile(.9):.1f}, p95 {mv.dstar.quantile(.95):.1f}, p99 {mv.dstar.quantile(.99):.1f}, p99.5 {mv.dstar.quantile(.995):.1f}")
print(f"half-normal prediction: p95 {1.960*sigma:.1f}, p99 {2.576*sigma:.1f}, p99.5 {2.807*sigma:.1f}  -> real tail is heavier than Gaussian")
# fixed radius table
rows = []
for r in range(5, 151, 5):
    for name, g in [("train", tr), ("student_test", st)]:
        cov = (g.dstar <= r).mean()
        rows.append({"radius_m": r, "set": name, "coverage": cov, "miss_rate": 1 - cov,
                     "mean_K": g[f"K{r}"].mean(), "p95_K": g[f"K{r}"].quantile(.95),
                     "frac_K0": (g[f"K{r}"] == 0).mean(), "mean_K2": (g[f"K{r}"] ** 2).mean()})
F = pd.DataFrame(rows)
T = F[F.set == "train"].set_index("radius_m")
T["gain_next_5m_pp"] = (T.coverage.shift(-1) - T.coverage) * 100
print("\nFIXED RADIUS (train):"); print(T[["coverage", "miss_rate", "gain_next_5m_pp", "mean_K", "p95_K", "frac_K0", "mean_K2"]].round(4).to_string())
# where are the misses at 30/40/50 m?  (which trips)
for r in (30, 40, 50):
    m = tr[tr.dstar > r]
    top = m.trip_id.value_counts().head(5)
    print(f"\nmisses at {r} m: {len(m)} pts; top trips {top.to_dict()}; share from trips 54,33,2: {m.trip_id.isin([54,33,2]).mean():.2f}")
# per-trip coverage stability (train)
for r in (30, 40, 50, 60):
    pc = tr.groupby("trip_id").apply(lambda g: (g.dstar <= r).mean())
    print(f"r={r}: per-trip coverage min {pc.min():.3f} (trip {pc.idxmin()}), 5th pct {pc.quantile(.05):.3f}, median {pc.median():.4f}")
# adaptive: candidates within min(dnear + delta, 150)
rows = []
for dl in (5, 10, 15, 20, 25, 30, 40, 50):
    for name, g in [("train", tr), ("student_test", st)]:
        cov = (g.dstar <= np.minimum(g.dnear + dl, 150)).mean()
        rows.append({"delta_m": dl, "set": name, "coverage": cov, "mean_K": g[f"KA{dl}"].mean(), "p95_K": g[f"KA{dl}"].quantile(.95),
                     "frac_K0": (g[f"KA{dl}"] == 0).mean(), "mean_K2": (g[f"KA{dl}"] ** 2).mean()})
A = pd.DataFrame(rows)
print("\nADAPTIVE (d_nearest + delta, cap 150 m):"); print(A[A.set == "train"].round(4).to_string(index=False))
F.to_csv("out/fixed_radius_table.csv", index=False); A.to_csv("out/adaptive_radius_table.csv", index=False)
