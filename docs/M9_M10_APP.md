# M9 (FastAPI backend) + M10 (Streamlit UI): built ahead of the model, with placeholders
Repo layout (commit as-is at the repo root):
```
pathsnap/        shared package v0.3.1 (hmm.TripLattice.from_parts added for the API)
api/             settings.py · artifacts.py (model loader + PLACEHOLDER) · service.py (prediction, never reads .route)
                 evaluator.py (server-side accuracy) · handlers.py (plain functions) · main.py (FastAPI)
ui/              app.py (Streamlit) · api_client.py · map_view.py (Folium)
scripts/         export_model.py (M8) · smoke_test_api.py · test_service_offline.py
tests/           http_test_server.py (same handlers via Starlette, for testing without FastAPI)
artifacts/       README.md (contract) — model.joblib / manifest.json / hmm_config.json go here
logs/            api.log + requests.csv (written by the API)
requirements_api.txt · requirements_ui.txt · DEPLOYMENT.md
```
Placeholders and how they disappear:
| Dependency not ready | What runs meanwhile | Switch-over |
|---|---|---|
| trained model | PLACEHOLDER rule scorer (flagged everywhere) | put model.joblib + manifest.json in artifacts/, restart the API |
| decoder config | M7 defaults β=3, break_s=300, γ=0 | put hmm_config.json in artifacts/, restart |
| evaluator trips | student-test trips | set PATHSNAP_EVAL_DIR to the evaluator folder, restart |

Tested in Claude's sandbox (no FastAPI/Streamlit there):
- service + evaluator on all 10 student-test trips: placeholder pooled Top-1 0.9085; dev-exported real model 0.9344 / 0.8668,
  identical to the M7 notebook → the swap is exact. 0.1–1.5 s per trip.
- HTTP + JSON: the same handlers served by Starlette; scripts/smoke_test_api.py passed (10 trips, 404 on a bad id, no reference
  data in responses); UI client + map + GeoJSON tested against it; logging format checked.
- NOT testable there: the FastAPI decorators and the Streamlit page rendering → run smoke_test_api.py on Laptop 1 and open
  the UI once on Laptop 2.
