"""PathSnap evaluator UI — Laptop 2 (paya5am).   Run:  streamlit run ui/app.py
Set the API address in the sidebar or with the env var PATHSNAP_API_URL (e.g. http://192.168.1.23:8000).
The UI never loads data or models: it lists the evaluation trips served by the API, sends the chosen trip id,
and shows the returned predicted route and the accuracy that the API computed with its internal reference data."""
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.api_client import health, trips, predict, ApiError
from ui.map_view import build_map, route_geojson

st.set_page_config(page_title="PathSnap — map matching", layout="wide")
st.title("PathSnap: GPS route reconstruction")
st.caption("Pick an evaluation trip. Laptop 1 runs the full pipeline (candidates → classifier → HMM/Viterbi), computes "
           "accuracy against its internal reference, and returns the predicted route. Ground truth never leaves Laptop 1.")

# ---------------- sidebar: connection ----------------
with st.sidebar:
    st.header("API connection")
    api = st.text_input("API URL (Laptop 1)", value=st.session_state.get("api", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")))
    st.session_state["api"] = api
    if st.button("Check connection") or "health" not in st.session_state:
        try:
            st.session_state["health"] = health(api); st.session_state.pop("health_error", None)
        except ApiError as e:
            st.session_state["health"] = None; st.session_state["health_error"] = str(e)
    h = st.session_state.get("health")
    if h:
        st.success(f"Connected · {h['n_eval_trips']} evaluation trips")
        if h["model_status"] == "placeholder":
            st.error("PLACEHOLDER model active on the API: these are NOT the final model's results.")
        else:
            st.info(f"Model: **{h['model_name']}**")
        st.write("Decoder:", h["decoder"], f"({h['decoder_status']})")
        for w in h.get("warnings", []):
            st.warning(w)
    else:
        st.error(st.session_state.get("health_error", "Not connected"))

if not st.session_state.get("health"):
    st.stop()

# ---------------- trip selection ----------------
try:
    trip_list = trips(api)
except ApiError as e:
    st.error(str(e)); st.stop()
if not trip_list:
    st.warning("The API has no evaluation trips. Set PATHSNAP_EVAL_DIR on Laptop 1."); st.stop()

st.subheader("1 · Choose an evaluation trip")
labels = {f"Trip {t['trip_number']}  ·  {t['gps_points']:,} GPS points": t["trip_id"] for t in trip_list}
choice = st.radio("Evaluation trips served by the API", list(labels), horizontal=True, label_visibility="collapsed")
run = st.button("Run map matching on the API", type="primary")

if run:
    with st.spinner(f"Laptop 1 is processing trip {int(labels[choice])} …"):
        try:
            st.session_state["result"] = predict(api, labels[choice])
            st.session_state.setdefault("history", []).append(st.session_state["result"])
        except ApiError as e:
            st.error(str(e))

res = st.session_state.get("result")
if res:
    st.subheader(f"2 · Result for trip {int(res['trip_id'])}")
    if res["model_status"] == "placeholder":
        st.error("Computed with the PLACEHOLDER scorer, not the trained model.")
    m = res["metrics"]
    c = st.columns(6)
    if m.get("available"):
        c[0].metric("Top-1 accuracy (HMM)", f"{m['top1_accuracy_hmm']:.2%}",
                    f"{m['top1_accuracy_hmm'] - m['top1_accuracy_classifier_only']:+.2%} vs classifier only")
        c[1].metric("Classifier only", f"{m['top1_accuracy_classifier_only']:.2%}")
        c[2].metric("Route-arc F1", f"{m['route_arc_f1']:.3f}")
    else:
        c[0].info("No reference route on the API for this trip, so accuracy is not available")
    c[3].metric("Route continuity", f"{m['route_continuity_reachable']:.2%}", f"strict {m['route_continuity_strict']:.2%}", delta_color="off")
    c[4].metric("GPS points", f"{res['n_points']:,}", f"{m.get('labelled_points', 0):,} scored", delta_color="off")
    c[5].metric("API time", f"{res['timings_s']['total_s'] + res['timings_s']['evaluation_s']:.2f} s")
    components.html(build_map(res).get_root().render(), height=620)
    with st.expander("Details: timings, decoder, model"):
        st.json({"model": res["model_name"], "model_status": res["model_status"], "decoder": res["decoder"],
                 "decoder_status": res["decoder_status"], "segments": res["n_segments"],
                 "points_without_prediction": res["points_without_prediction"], "candidate_rows": res["n_candidate_rows"],
                 "timings_s": res["timings_s"], "metrics": m, "warnings": res["warnings"]})
    d1, d2 = st.columns(2)
    d1.download_button("Download predicted route (GeoJSON)", route_geojson(res), file_name=f"trip_{res['trip_id']}_route.geojson")
    d2.download_button("Download per-point predicted arcs (CSV)",
                       pd.DataFrame({"gps_point": range(res["n_points"]), "lat": [p[0] for p in res["gps"]],
                                     "lon": [p[1] for p in res["gps"]], "predicted_arc": res["predicted_arc_per_point"]}).to_csv(index=False),
                       file_name=f"trip_{res['trip_id']}_points.csv")

if st.session_state.get("history"):
    st.subheader("3 · Runs in this session")
    st.dataframe(pd.DataFrame([{"trip": int(r["trip_id"]), "model": r["model_status"],
                                "Top-1 HMM": r["metrics"].get("top1_accuracy_hmm"),
                                "Top-1 classifier": r["metrics"].get("top1_accuracy_classifier_only"),
                                "route-arc F1": r["metrics"].get("route_arc_f1"),
                                "continuity": r["metrics"]["route_continuity_reachable"],
                                "seconds": round(r["timings_s"]["total_s"], 2)} for r in st.session_state["history"]]),
                 )
