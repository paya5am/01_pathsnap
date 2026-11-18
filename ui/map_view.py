"""Folium map of one API response: raw GPS points + predicted (HMM) route. No reference route: ground truth stays on
the API laptop."""
import json
import folium


def build_map(resp, max_points=4000):
    gps = resp["gps"]; route = resp["predicted_route"]
    lats = [p[0] for p in gps]; lons = [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14,
                   tiles="cartodbpositron", control_scale=True)
    g_route = folium.FeatureGroup(name="Predicted route (HMM + Viterbi)")
    for seg in route:
        folium.PolyLine(seg, color="#1f4fd8", weight=5, opacity=0.85).add_to(g_route)
    g_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(gps) // max_points)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=2, color="#e4572e", fill=True, fill_opacity=0.8, weight=0,
                            popup=f"GPS point {i}").add_to(g_gps)
    g_route.add_to(m); g_gps.add_to(m)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #999;border-radius:6px;font-size:13px"><b>Trip {int(resp['trip_id'])}</b><br>
      <span style="color:#1f4fd8">&#9473;&#9473;</span> predicted route<br>
      <span style="color:#e4572e">&#9679;</span> raw GPS ({len(gps):,} points)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


def route_geojson(resp):
    feats = [{"type": "Feature", "properties": {"kind": "predicted_arc", "order": i},
              "geometry": {"type": "LineString", "coordinates": [[p[1], p[0]] for p in seg]}}
             for i, seg in enumerate(resp["predicted_route"])]
    return json.dumps({"type": "FeatureCollection", "features": feats})
