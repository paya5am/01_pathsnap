# M2 — Dataset exploration: summary (notebook 01_eda; owner A)
Partitions analysed: train (80) + student_test (10); evaluator trips unavailable and untouched.
- Files per trip: .track (lon, lat, t; ~1 Hz), .nodes (lon, lat), .arcs (directed from→to), .route (ordered reference arcs —
  NOT one per GPS point: median 5.25 points per route arc). Each trip ships its own road network (649 – 307k arcs).
- Values: 0 missing, 0 out-of-range, timestamps strictly increasing; 96.7% of steps exactly 1 s; gaps > 60 s in 23 trips,
  > 1 h in trips 78 (14.7 h) and 90 (42.5 h).
- Network: 95% of arcs two-way; 1,903 duplicate arcs, 24 self-loops, 92 nodes sharing coordinates → 106 zero-length arcs;
  route arcs outside the largest strongly-connected component in 10 trips.
- Reference route: breaks in 11 trips; trip 2 ends early; trips 54/33 parked off-network ~79% of the time; trip 66 has a
  ~65 s segment pointing against the driving direction; trip 11 is out-and-back.
- Validity checks V1–V7 (section 13): lon/lat order confirmed; seconds confirmed (fractional offsets in trips 2/71/97);
  no duplicated trips; no train↔student-test overlap; trips 10 & 16 overlap (same CV group).
- Projection: per-trip local equirectangular, max error 0.72 m on consecutive points vs haversine.
- Candidate rule (section 10, train-only selection): arcs within min(d_nearest + 20 m, 150 m) → clean-train coverage 98.47%,
  4.9 candidates per point, never empty (vs fixed 50 m: 98.3% with 9.5 candidates).
- Nearest arc is on the route 86.5% ignoring direction but 60.2% with direction → heading is essential.
