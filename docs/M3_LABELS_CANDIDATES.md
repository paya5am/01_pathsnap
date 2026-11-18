# M3 — Labels, candidates, dataset: summary (notebook 02; owners A labels/candidates, B features)
- Labels: monotone dynamic-programming alignment of GPS points to route positions (never backwards in time), cost
  min(dist, 200 m) + 100 m × [moving and arc points against the motion]. Unlabelled if the matched arc is > 50 m away or in a
  heading-mismatch run of ≥ 10 points. Labelled: train 88.81%, student-test 99.86%; 92.5% of unlabelled points come from
  trips 54/33/2. Robustness: ≥ 99.73% same arc under every setting change; naive nearest-route-arc labels would differ on 5.49%.
- Candidates: arcs within min(d_nearest + 20 m, 150 m), zero-length dropped, duplicates merged; candidate miss rate on labelled
  points: train 0.77%, student-test 0.30%.
- Dataset: 1,155,525 (point, candidate) rows, 16 features (docs/FEATURES_EXPLAINED.md), label 1/0/−1, cv_group (trip; 16→10).
- Rule baselines (student-test Top-1): nearest arc 0.533; nearest arc pointing the car's way 0.835.
