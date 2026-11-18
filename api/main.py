"""PathSnap API — Laptop 1 (backend: model, HMM, server-side evaluation).

Run from the repo root:   python -m api.main          (or: uvicorn api.main:app --host 0.0.0.0 --port 8000)
Endpoints:
  GET  /health     model/decoder status (trained vs PLACEHOLDER), versions, warnings
  GET  /trips      evaluation trips available on this laptop (ids + GPS point counts; no ground truth)
  POST /predict    {"trip_id": "15"} → full pipeline → predicted route + accuracy computed here, server-side
Every request is logged to this console (show it to the evaluator), to logs/api.log and to logs/requests.csv.
"""
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from api import settings
from api.artifacts import load_artifacts
from api.handlers import health_payload, trips_payload, predict_payload, list_eval_trips

settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(settings.LOG_DIR / "api.log", encoding="utf-8")])
log = logging.getLogger("pathsnap.api")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE["artifacts"] = load_artifacts(settings.ARTIFACT_DIR, settings.ALLOW_PLACEHOLDER)
    trips = list_eval_trips()
    log.info("evaluation folder: %s | %d trips: %s", settings.EVAL_DIR, len(trips), [int(t) for t in trips])
    if not trips:
        log.warning("No trips in %s: set PATHSNAP_EVAL_DIR to the folder that contains the trip folders.", settings.EVAL_DIR)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d in %.0f ms", request.method, request.url.path, client, response.status_code,
             1000 * (time.perf_counter() - t0))
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["/health", "/trips", "/predict"]}


@app.get("/health")
def health():
    return health_payload(STATE["artifacts"])


@app.get("/trips")
def trips():
    return trips_payload()


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_payload(req.trip_id, STATE["artifacts"])
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
