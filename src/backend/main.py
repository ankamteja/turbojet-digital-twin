"""FastAPI service exposing the digital twin to the dashboard.

Run:  uvicorn main:app --reload --port 8000   (from src/backend/)
"""

from __future__ import annotations

# Cap the numerical thread pools BEFORE importing sklearn/joblib. The startup
# precompute fires many small gradient-boosting predictions; without this the
# OpenMP/BLAS pools oversubscribe the CPU and startup stalls at high load.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_v, "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import EngineInfo, EngineState, SensorRow
from service import TwinService

app = FastAPI(title="Turbojet Digital Twin API", version="1.0")

# open CORS for local frontend dev (static server on :8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# built once at startup — loads the ensemble and precomputes engine states
twin: TwinService | None = None


@app.on_event("startup")
def _startup():
    global twin
    twin = TwinService()


@app.get("/api/health")
def health():
    return {"status": "ok", "engines": len(twin.engines()) if twin else 0}


@app.get("/api/engines", response_model=list[EngineInfo])
def engines():
    return twin.engines()


@app.get("/api/engine/{engine_id}/history")
def history(engine_id: int):
    try:
        return twin.history(engine_id)
    except KeyError:
        raise HTTPException(404, f"unknown engine {engine_id}")


@app.get("/api/engine/{engine_id}/state", response_model=EngineState)
def state(engine_id: int, cycle: int):
    try:
        return twin.state(engine_id, cycle)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/predict", response_model=EngineState)
def predict(row: SensorRow):
    return twin.predict_row(row.model_dump())


@app.get("/api/engine/{engine_id}/simulate")
def simulate(engine_id: int, to_cycle: int = 60):
    try:
        return twin.simulate(engine_id, to_cycle)
    except KeyError:
        raise HTTPException(404, f"unknown engine {engine_id}")
