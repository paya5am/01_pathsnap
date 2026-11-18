# M11 integration test report
model: hist_gb_tuned_ll (trained) · decoder {'beta': 3.0, 'break_s': 300.0, 'gap_exp': 0.0} · pathsnap 0.3.2

| Test | Result | Detail |
|---|---|---|
| T0 trained model loads (placeholder disabled) | PASS | hist_gb_tuned_ll, decoder {'beta': 3.0, 'break_s': 300.0, 'gap_exp': 0.0}, warnings [] |
| T1 prediction never uses the reference (.route removed → identical output) | PASS |  |
| T1b evaluation degrades gracefully without a reference | PASS |  |
| T2 deterministic (same trip twice → identical) | PASS |  |
| T3 end-to-end numbers equal notebook 07 (HMM / classifier-only) | PASS | HMM 0.9344 (expected 0.9344), classifier 0.8668 (expected 0.8668) |
| T4 latency: every trip predicted + evaluated < 5 s | PASS | max 1.51s, mean 0.66s |
| T5 response has no reference data, is JSON-serialisable, lengths consistent | PASS | 45 KB |
| T6 trip id accepted as '7' and '00000007' | PASS |  |
| T7 unknown trip id rejected | PASS |  |
| T8 /trips and /health payloads | PASS |  |
| T9 contract: wrong feature order refused; missing model refused when placeholders disabled; placeholder flagged | PASS |  |
| T10a hard trips run without error and < 10 s each | PASS | 90(42-hour gap): 536 pts, 2 segs, 0 no-pred, 0.2s; 78(15-hour gap): 2909 pts, 2 segs, 0 no-pred, 2.1s; 21(15,783 points): 15783 pts, 3 segs, 0 no-pred, 1.3s; 54(parked off-network): 19313 pts, 57 segs, 0 no-pred, 4.3s; 2(route ends early / off-network): 7243 pts, 6 segs, 191 no-pred, 1.6s; 43(GPS drift on divided motorway): 2322 pts, 137 segs, 7 no-pred, 2.6s; 11(out-and-back): 2828 pts, 2 segs, 0 no-pred, 0.4s; 75(reference route breaks): 1628 pts, 3 segs, 0 no-pred, 0.3s; 13(sustained >60 m/s): 2580 pts, 1 segs, 0 no-pred, 3.6s |
| T10b decoded-route reachability ≥ 0.99 on hard trips, except KNOWN LIMITATION trips [43] | PASS | below 0.99: [(43, 0.9416)] (trip 43 = documented M7 limitation: drift → chain cuts) |