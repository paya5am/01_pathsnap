# PathSnap — the 16 model features, explained (M3, decision D15; reviewed in M8)
Numbers: notebook 02 table 09_feature_separation_train.csv (TRAIN rows only). "Separation" = single-feature ROC-AUC folded
to ≥ 0.5 (0.5 = no signal). Each dataset row is one (GPS point, candidate arc) pair; for each point the highest-scoring
candidate wins.
- **Candidate features** differ between the candidates of one point → decide WHICH arc wins.
- **Context features** are equal for all candidates of a point → tell the model how much to trust the others.

| Group | Feature | Definition | Why (evidence) | Separation |
|---|---|---|---|---|
| Distance | `dist` | point-to-segment distance (m) | GPS noise σ ≈ 7.5 m (M2) | 0.788 |
| | `dist_rel` | dist − distance to the nearest candidate | parallel carriageways ~25 m apart (M2) | 0.865 |
| | `dist_ratio` | (dist+1)/(d_nearest+1) | scale-free relative distance | 0.857 |
| | `dist_rank` | 0 = nearest candidate, 1 = second… | ordinal, robust | 0.865 |
| | `win_dist_mean` | mean distance of points i−1, i, i+1 (±1 s, gap-aware) to this arc | smooths jitter; ±1 s beat ±2/3/5 s (0.817/0.808/0.783/0.724) | 0.817 |
| Direction | `cos_heading` | cos(GPS heading − arc bearing); 0 if unknown | 95% of arcs are two-way twins: only direction separates them (M2) | 0.839 |
| | `heading_known` | moving ≥ 3 m/s, window not spanning a gap | heading is noise when slow | context |
| Geometry | `dist_to_end_m` | min(t, 1−t) × arc length | a wrong neighbouring arc touches the point only at the shared junction | 0.839 |
| | `arc_len_log` | log(1 + length) | arcs 0.01 m – 9.3 km | 0.725 |
| | `two_way` | reverse arc exists | twin competing | 0.534 |
| | `from_out_deg`, `to_out_deg` | arcs leaving start / end node | junction complexity | 0.53 |
| Context | `n_cands`, `d_nearest`, `speed`, `turn` | candidate count; distance to nearest arc; speed over ±2 s (clipped 60 m/s); heading change over ±4 s | density, local GPS quality, motorway vs town, junctions | context |

Dropped in M3 (train-only evidence): `heading_diff` (= cos_heading, r = −1.00), `win_dist_max` (r = 0.91 with win_dist_mean),
`win_n`, `proj_t`, `dt_prev`, `dt_next` (single-feature AUC ≈ 0.50).

## M8 review (notebook 08): C's extra features
C's LightGBM pipeline used `proj_t` (position along the arc), `dist_prev3`/`dist_next3` (distance of the fix 3 samples
earlier/later to this arc), raw heading difference and heading × moving(≥2 m/s). Adding these 5 to our 16, same model and folds:
classifier OOF Top-1 0.8521 → 0.8832 (+3.1 points). Lesson: **dropping features by single-feature AUC is unreliable for
features that only work in combination** (proj_t alone 0.508). But after HMM decoding the gain disappears (0.9246 → 0.9211
pooled; per-trip mixed): these features carry temporal context that the HMM already provides. The deployed model therefore
keeps the 16 approved features; the 5 extra columns are still computed (`pathsnap.features.C_EXTRA`) for reproducibility.

No leakage: every feature uses only the GPS track + road network (identical with and without the .route file). Nothing is fitted
in M3; scalers are fitted inside model pipelines on training folds only.
