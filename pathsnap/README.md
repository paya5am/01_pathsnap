# pathsnap — shared preprocessing package (M3)
Used by every notebook AND by the API, so training and evaluation run identical code.

| module | owner | purpose |
|---|---|---|
| config.py | all | every tunable constant, each with its evidence reference |
| io.py | A | read a trip folder (.track/.nodes/.arcs/.route) |
| geo.py | A | per-trip metre projection, bearings, point-to-segment projection |
| network.py | A | projected network: drop zero-length arcs, merge duplicate arcs, two-way flag, degrees, KD-tree index |
| track.py | B | per-point motion: heading/speed (±2 s), moving flag (gap-aware), turn angle |
| candidates.py | A | candidates = arcs within min(d_nearest + 20 m, 150 m) |
| features.py | B | 16 features per (point, candidate); FEATURES = model input list |
| labels.py | A | heading- and order-aware DP alignment of GPS points to the ordered route → per-point reference arc + status |
| pipeline.py | A+B | process_trip(trip_dir, with_labels) → (candidate feature rows, per-point table, internals) |

Quick use:
```python
import sys; sys.path.insert(0, "<repo root>")
from pathsnap.pipeline import process_trip
feats, points, _ = process_trip("data/train/00000001")              # training (with labels)
feats, points, _ = process_trip("data/eval/00000015", with_labels=False)   # evaluation: route never read
```
Requirements: numpy, pandas, scipy, scikit-learn (notebook analysis), matplotlib, folium; pyarrow optional (parquet).
