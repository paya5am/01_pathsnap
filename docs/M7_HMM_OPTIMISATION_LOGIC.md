# M7 — HMM + Viterbi: model, optimisation logic and results
Code: `pathsnap/hmm.py` · notebook `07_hmm_viterbi` · owner C (nisarg-ravi). Numbers: `experiments/results/m7/`.

## 1. Why a sequence model at all
A per-point classifier decides every GPS point independently. Two failures follow (M4 error analysis):
1. **Stopped points:** a parked fix lies equally on both directions of a two-way road → classifier Top-1 only 0.43 when stopped.
2. **Impossible routes:** consecutive picks jump between unconnected roads → only 93% of consecutive classifier picks are
   directly connected (vs 98.5% for the reference labels).
An HMM scores whole *sequences* of arcs, so the decision at time t uses the past and the future of the trip.

## 2. The model
| Element | Definition | Reason |
|---|---|---|
| Hidden state at point t | one of the candidate arcs of point t (rule: arcs within min(d_nearest + 20 m, 150 m); ~5 per point) | keeps the candidate ↔ arc mapping used in training; small state space |
| Emission score | `log P(arc correct | features)` from the classifier (hist_gb_tuned_ll) | the classifier already combines distance, heading and geometry evidence; it was chosen partly for calibration |
| Transition score a → b | `−| d_route(a→b) − d_gps(t,t+1) | / β` | Newson & Krumm (2009): the distance a car really drives between two fixes ≈ the straight-line distance between them |
| d_route | projection on a → end of a → **directed shortest path** (Dijkstra on the road graph) → start of b → projection on b | directed graph = one-way streets are respected; a U-turn onto the twin costs a detour, so a car keeps its direction |
| Same arc | forward distance along it; moving backwards ≤ 15 m allowed | GPS jitter while crawling must not force arc changes |
| Impossible transition | b not reachable within 3·d_gps + 200 m → score −∞ | continuity constraint; also bounds Dijkstra |
| Gaps / dead ends | Δt > break_s, or no candidate of t+1 reachable from any state of t → cut the chain, decode the next segment separately | trips with 15 h / 42 h gaps; route breaks and disconnected network parts (M2) |
| No candidates | point gets no prediction, chain cut | never happened on labelled points |

**Viterbi recurrence (log space):**
δ₁(j) = log e₁(j)
δₜ₊₁(j) = log eₜ₊₁(j) + maxᵢ [ δₜ(i) − |d_route(i→j) − d_gps(t)| / β ],  ψₜ₊₁(j) = argmaxᵢ(…)
Final path: j*_T = argmax_j δ_T(j), then j*ₜ = ψₜ₊₁(j*ₜ₊₁) backwards. Exact most-likely path, O(T·K²) per trip (K ≈ 5).
The decoded arcs are then joined with shortest paths between consecutive different arcs → one connected route (API/UI).

## 3. What was optimised, and how (all on the 80 TRAINING trips; student-test used once at the end)
**Leakage rule:** decoder parameters were tuned on *out-of-fold* probabilities (each training trip scored by a classifier that
never saw it), so the decoder is tuned on the same kind of probabilities it gets at test time.

**Step 1 — emission model.** M6 left three hist_gb configurations tied on classifier accuracy. Each was decoded on the training
trips (small β grid) and compared on Top-1 *after* the HMM (the system objective):
| Emission model | OOF Top-1 classifier | OOF Top-1 after HMM | CV log loss |
|---|---|---|---|
| hist_gb_tuned (best Top-1 trial) | 0.8527 | 0.9250 | 0.1258 |
| hist_gb (untuned) | 0.8517 | 0.9243 | 0.1252 |
| hist_gb_tuned_ll (best log loss) | 0.8521 | 0.9241 | **0.1212** |
All within 0.001 → rule fixed in advance: lowest log loss → **hist_gb_tuned_ll** (most honest probabilities, most stable folds).

**Step 2 — decoder parameters, grid search (120 settings):** β ∈ {0.25, 0.5, 1, 2, 3, 5, 10, 20, 40, 80} m ×
break_s ∈ {30, 60, 120, 300} s × γ ∈ {0, 0.5, 1} (gap scaling β·Δt^γ). Objective: pooled Top-1 on labelled training points.
| β (m), break 300 s, γ 0 | 0.25 | 0.5 | 1 | 2 | **3** | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train OOF Top-1 | 0.908 | 0.914 | 0.920 | 0.924 | **0.9246** | 0.922 | 0.919 | 0.918 | 0.912 | 0.906 |
- **β = 3 m** is an interior optimum (the first grid stopped at 2 m with the best value on its edge, so it was extended).
  Small β trusts the road network too much (any GPS/route-length mismatch is punished and the classifier is overruled);
  large β ignores connectivity and the decoder falls back towards classifier-only. 2–5 m is a plateau (within 0.003).
- **γ = 0** (no extra slack across gaps) beats 0.5 / 1 (0.9246 vs 0.9236 / 0.9225): with 1-s sampling most steps are 1 s; extra
  slack mainly lets the decoder jump to wrong roads after short gaps.
- **break_s = 300 s** (0.9246) ≈ 60–120 s (0.9245), slightly better than 30 s (0.9238): cutting the chain too eagerly throws
  away useful context; very long gaps (15 h, 42 h) are cut anyway.
- Fixed by reasoning, not tuned: back tolerance 15 m (≈ 2σ of GPS noise), reach limit 3·d_gps + 200 m.

## 4. Results
| Student-test (10 trips) | Top-1 | Continuity (strict) | Continuity (reachable) | Route-arc F1 |
|---|---|---|---|---|
| Classifier only | 0.8668 | 0.9316 | 0.9820 | 0.9111 |
| **Classifier + HMM** | **0.9344** | **0.9830** | **0.9997** | **0.9666** |
Training trips (OOF): 0.8521 → 0.9246. Where the gain comes from (student-test): stopped/slow points 0.427 → 0.728; moving
0.942 → 0.970; twin errors 6.4% → 3.0%; 1,631 points fixed, 339 made worse. All 10 student-test trips improve; 76 of 80
training trips improve. Speed: decoding all 10 student-test trips takes 0.26 s (lattice building ~0.2–0.5 s per trip).

## 5. Where it does not help (honest limits)
- Trip 10 (−5.6 points): parked 45% of the time in dense Moscow streets; the decoder commits to a wrong chain through jitter.
- Trips 57, 97, 5 (−0.1 to −1.7 points): slow U-turns: the decoder keeps one direction while the reference switches.
- Trip 43: GPS drifts > 20 m onto the opposite motorway carriageway, the true arc is outside the candidate window and nothing is
  reachable → 136 chain cuts (still +3.5 points overall; reachability 0.94, the M11 known limitation).
  Improvement for later: skip outlier points and bridge over them instead of cutting.

## 6. Plain words
The decoder looks at the whole trip and asks: "which chain of road pieces could a car really have driven to produce these dots?"
Each road piece gets a score from the classifier (does the dot look like it's on this road?) and each step between pieces gets a
score from the map (could the car drive from here to there in one second, about as far as the dots moved?). Viterbi finds the
best-scoring chain exactly. One knob (β) decides how strict the map is; we tried 10 values and 3 m worked best on training trips.
Result: 87% → 93% correct on unseen trips, and a route you could actually drive.
