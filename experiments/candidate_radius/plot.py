import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv("out/radius_points.csv.gz"); ANOM = [54, 33, 2]
trc = d[(d.partition == "train") & ~d.trip_id.isin(ANOM)]; st = d[d.partition == "student_test"]
R = list(range(5, 151, 5)); D = [5, 10, 15, 20, 25, 30, 40, 50]
fx = [((trc.dstar <= r).mean(), trc[f"K{r}"].mean()) for r in R]
ad = [((trc.dstar <= np.minimum(trc.dnear + dl, 150)).mean(), trc[f"KA{dl}"].mean()) for dl in D]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot([k for _, k in fx], [c for c, _ in fx], "o-", label="fixed radius r", color="#1f77b4", ms=4)
ax[0].plot([k for _, k in ad], [c for c, _ in ad], "s-", label="adaptive: d_nearest + δ", color="#d62728", ms=5)
for r, (c, k) in zip(R, fx):
    if r in (15, 25, 50, 100): ax[0].annotate(f"r={r}", (k, c), textcoords="offset points", xytext=(5, -12), fontsize=8, color="#1f77b4")
for dl, (c, k) in zip(D, ad):
    if dl in (10, 20, 30, 50): ax[0].annotate(f"δ={dl}", (k, c), textcoords="offset points", xytext=(-10, 6), fontsize=8, color="#d62728")
c20, k20 = ad[D.index(20)]; ax[0].scatter([k20], [c20], s=160, facecolors="none", edgecolors="k", lw=1.5, zorder=5)
ax[0].set(xlim=(0, 25), ylim=(0.80, 1.0), xlabel="mean candidates per GPS point (cost)", ylabel="coverage = 1 − candidate miss rate",
          title="Coverage vs cost — train trips (excl. anomalous 54/33/2)"); ax[0].grid(alpha=.3); ax[0].legend(loc="lower right")
for g, name, sty in [(trc, "train (clean)", "s-"), (st, "student-test (check only)", "^--")]:
    ax[1].plot(D, [(g.dstar <= np.minimum(g.dnear + dl, 150)).mean() for dl in D], sty, label=name)
ax[1].axvline(20, ls=":", c="k"); ax[1].set(xlabel="δ (m) beyond the nearest arc", ylabel="coverage", ylim=(0.9, 1.0),
          title="Adaptive rule: coverage vs δ (chosen δ = 20 m)"); ax[1].grid(alpha=.3); ax[1].legend(loc="lower right")
plt.tight_layout(); plt.savefig("out/radius_choice.png", dpi=120)
