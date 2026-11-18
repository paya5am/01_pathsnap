"""Test-only HTTP server: exposes the SAME handlers as api/main.py using Starlette (which FastAPI is built on).
Used where FastAPI isn't installed, to test the HTTP + JSON flow end-to-end with the real pipeline."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from api import settings
from api.artifacts import load_artifacts
from api.handlers import health_payload, trips_payload, predict_payload
ART = load_artifacts(settings.ARTIFACT_DIR, True)

async def health(request): return JSONResponse(health_payload(ART))
async def trips(request): return JSONResponse(trips_payload())
async def predict(request):
    body = await request.json()
    try: return JSONResponse(predict_payload(body["trip_id"], ART))
    except LookupError as e: return JSONResponse({"detail": str(e)}, status_code=404)
app = Starlette(routes=[Route("/health", health), Route("/trips", trips), Route("/predict", predict, methods=["POST"])])
