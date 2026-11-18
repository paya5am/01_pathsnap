"""Build the candidate table for all train + student_test trips and per-trip edge-case stats.

usage: python scripts/build_dataset.py <data_root> <out_dir>
Writes <out_dir>/candidates_<partition>.npz and <out_dir>/trip_edge_cases.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import io, pipeline  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402


def edge_stats(trip, prep):
    net = prep["net"]; t = prep["t"]; r = trip.route
    dt = np.diff(t)
    breaks = int(np.sum(net.to[r[:-1]] != net.frm[r[1:]]))
    n = len(net.nx)
    use = net.frm != net.to
    g = csr_matrix((np.ones(use.sum()), (net.frm[use], net.to[use])), shape=(n, n))
    _, comp = connected_components(g, directed=True, connection="strong")
    big = np.bincount(comp).argmax()
    in_scc = float(np.mean((comp[net.frm[r]] == big) & (comp[net.to[r]] == big)))
    lab = prep["label"]
    P = prep["cand"]["point"]; y = prep["y"]
    hit = np.zeros(len(lab), bool); hit[P[y == 1]] = True
    labelled = lab >= 0
    return dict(
        trip_id=trip.trip_id, n_points=len(t), n_route_arcs=len(r), n_arcs=len(net.frm),
        gaps_gt60s=int(np.sum(dt > 60)), max_gap_s=float(dt.max()) if len(dt) else 0.0,
        route_breaks=breaks, route_in_largest_scc=round(in_scc, 4),
        revisited_route_arcs=int(len(r) - len(np.unique(r))),
        parked_frac=round(float(np.mean(prep["speed"] < 0.5)), 4),
        unlabelled_points=int((~labelled).sum()),
        cand_per_point=round(len(P) / len(t), 3),
        cand_miss_rate_labelled=round(float(1 - hit[labelled].mean()) if labelled.any() else np.nan, 5),
    )


def main(root, out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for part in ["train", "student_test"]:
        Xs, ys, gs, ps, arcs, lab_pts = [], [], [], [], [], []
        for tid in io.list_trips(root, part):
            t0 = time.time()
            trip = io.load_trip(root, part, tid)
            prep = pipeline.prepare_trip(trip)
            P = prep["cand"]["point"]
            Xs.append(prep["X"]); ys.append(prep["y"]); gs.append(np.full(len(P), tid, np.int32))
            ps.append(P.astype(np.int32)); arcs.append(prep["cand"]["arc"].astype(np.int64))
            lab_pts.append((prep["label"] >= 0)[P])
            s = edge_stats(trip, prep); s["partition"] = part; s["prep_s"] = round(time.time() - t0, 2)
            rows.append(s)
            print(part, tid, s["n_points"], s["cand_miss_rate_labelled"], s["prep_s"], flush=True)
        np.savez_compressed(os.path.join(out, f"candidates_{part}.npz"),
                            X=np.concatenate(Xs), y=np.concatenate(ys), trip=np.concatenate(gs),
                            point=np.concatenate(ps), arc=np.concatenate(arcs),
                            point_labelled=np.concatenate(lab_pts), features=np.array(FEATURES))
    pd.DataFrame(rows).to_csv(os.path.join(out, "trip_edge_cases.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
