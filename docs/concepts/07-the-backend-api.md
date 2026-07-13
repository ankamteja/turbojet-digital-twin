# 7. The Backend API

Files 1–6 covered the *model* — how sensor readings become health, performance,
confidence and RUL. But the model is a set of Python functions. The dashboard
(file 8) is a web page running in a browser. Those two live in different worlds
and can't call each other directly. The **backend API** is the bridge between
them.

The code is in `src/backend/`: `schemas.py`, `service.py`, and `main.py`.

## What is a web API, in plain language?

An **API** (Application Programming Interface) is a menu of requests one program
can make to another. A **web API** does this over the internet's own language,
HTTP — the same protocol your browser uses to load pages.

Analogy: a restaurant. You (the dashboard) don't walk into the kitchen (the
model) and cook. You read a menu, place an order, and a waiter brings back a
plate. The API is the menu-plus-waiter: a fixed set of things you may ask for,
and structured answers you get back.

A **REST API** is the most common style of web API. It's built around:

- **URLs** that name a resource, like `/api/engine/3/state` ("engine 3's state").
- **HTTP methods** that name the action: **GET** to fetch data, **POST** to send
  data.
- **JSON** as the format for answers — a simple, text-based way to write nested
  data (numbers, strings, lists, key/value objects) that both Python and
  JavaScript understand.

## FastAPI: the framework

We build the API with **FastAPI**, a Python library for writing web APIs. It lets
you turn a plain Python function into a web endpoint by tagging it with a
**decorator** — a line starting with `@` that says "run this function when a
request hits this URL." The whole app is created in `main.py`
(`src/backend/main.py:14`):

```python
app = FastAPI(title="Turbojet Digital Twin API", version="1.0")
```

Because the dashboard is served from a different address than the API, the
browser's security rules would normally block the calls. **CORS** middleware
opens that door for local development (`src/backend/main.py:17`):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # any origin may call us (fine for local dev)
    ...
)
```

## The contract: EngineState (schemas.py)

Before the endpoints, the most important idea in the backend: the **contract**.

A contract is a fixed, agreed-upon *shape* for the data that crosses the bridge.
The dashboard is written assuming every answer looks a certain way. As long as
the backend always produces that shape, the model behind it can change freely —
trees, neural nets, anything — and the dashboard never needs to know. The file
header says exactly this (`src/backend/schemas.py:1`):

> *"The frontend only ever sees these models. The model internals (features,
> ensemble size, losses) can change freely as long as EngineState stays stable."*

The shapes are defined with **Pydantic**, a library where you describe data as a
Python class and it validates every field's type for you. The central shape is
`EngineState` (`src/backend/schemas.py:69`):

```python
class EngineState(BaseModel):
    engine_id: int
    cycle: int
    conditions: Conditions
    sensors: Sensors
    health: dict[str, ComponentHealth]  # keys: compressor/combustor/turbine/overall
    performance: Performance
    alerts: list[Alert]
    history: list[HistoryPoint]
```

Read it top to bottom — it's the entire dashboard in one object:

- `conditions` — the flight situation (altitude, Mach, ambient temp/pressure,
  RPM, fuel flow).
- `sensors` — the raw gas-path readings (station pressures/temperatures, spool
  speed as a percent) for the HUD gauges.
- `health` — one `ComponentHealth` per subsystem *and* overall. Each carries the
  health `value`, `confidence`, `trend` (slope), `rul_cycles`, a
  `recommendation`, and the `contributing` sensor values behind the call
  (`src/backend/schemas.py:23`). This is exactly the model output from files 4–6.
- `performance` — thrust and TSFC, each with a value and a confidence.
- `alerts` — any warnings the backend raised.
- `history` — the health/thrust trend up to this cycle, for the trend chart.

Every field is a real model output or a physics-derived value. Nothing is faked.

## The endpoints (main.py)

Each endpoint is a Python function tagged with a FastAPI decorator naming its URL.
Here is the full menu.

**Liveness check** — is the server up? (`src/backend/main.py:34`)

```python
@app.get("/api/health")
def health():
    return {"status": "ok", "engines": len(twin.engines()) if twin else 0}
```

**List engines** — what engines exist and how many cycles each has
(`src/backend/main.py:39`):

```python
@app.get("/api/engines", response_model=list[EngineInfo])
def engines():
    return twin.engines()
```

**One engine's full history** — every cycle's predictions plus ground truth, for
charts (`src/backend/main.py:44`).

**A single state snapshot** — the heart of the dashboard. Note the `?cycle=`
query parameter: the URL asks for one engine at one moment in its life
(`src/backend/main.py:52`):

```python
@app.get("/api/engine/{engine_id}/state", response_model=EngineState)
def state(engine_id: int, cycle: int):
    try:
        return twin.state(engine_id, cycle)
    except KeyError as e:
        raise HTTPException(404, str(e))
```

The `response_model=EngineState` tells FastAPI to *enforce the contract* — if the
returned data didn't match the shape, it would error. If the requested engine or
cycle doesn't exist, it returns a clean `404 Not Found` rather than crashing.

**Ad-hoc prediction** — a **POST** endpoint: instead of asking about a stored
engine, you *send* one raw sensor row in the request body and get an
`EngineState` back (`src/backend/main.py:60`):

```python
@app.post("/api/predict", response_model=EngineState)
def predict(row: SensorRow):
    return twin.predict_row(row.model_dump())
```

**Forward simulation** — project an engine's health into the future, past the
last observed cycle, using the fitted degradation line
(`src/backend/main.py:65`):

```python
@app.get("/api/engine/{engine_id}/simulate")
def simulate(engine_id: int, to_cycle: int = 60):
    ...
```

File 8 shows how the dashboard's "projected degradation" drill-down uses this.

## Startup caching: doing the slow work once (service.py)

Running the model on demand for every web request would be wasteful — the same
engine's predictions never change. So the backend does all the heavy lifting
**once, at startup**, and then just serves cached answers.

FastAPI runs a startup hook when the server boots (`src/backend/main.py:28`):

```python
@app.on_event("startup")
def _startup():
    global twin
    twin = TwinService()
```

Building `TwinService` loads the trained ensemble and precomputes every engine's
per-cycle predictions and degradation fits (`src/backend/service.py:57`):

```python
class TwinService:
    def __init__(self):
        self.predictor = Predictor()
        ...
        for eid in sorted(self.raw["EngineID"].unique()):
            df_e = self.raw[self.raw["EngineID"] == eid]
            self._pred_cache[int(eid)] = self.predictor.predict_frame(df_e).sort_values("Cycle")
            self._traj_cache[int(eid)] = self.predictor.engine_trajectory(df_e)
```

After this, every `/state` request is a fast dictionary lookup, not a model run.
The dashboard feels instant even while scrubbing through cycles.

`TwinService` is also where the model's raw numbers become dashboard-friendly.
Two examples:

- **Spool speed as a percent.** The gauge wants 0–100%, so RPM is divided by the
  fleet's maximum RPM (`src/backend/service.py:74`).
- **Alerts.** Health values are turned into severity-tagged messages — "critical"
  below 0.85, "warn" below 0.90, an "info" note if the estimate is uncertain
  (`src/backend/service.py:42`).

## Recap

- A **REST API** is a fixed menu of URL-addressed requests returning **JSON**;
  it's the bridge from the Python model to the browser dashboard.
- **FastAPI** turns Python functions into endpoints; **CORS** lets the local
  dashboard call them.
- **`EngineState`** (Pydantic, in `schemas.py`) is the *contract* — a stable data
  shape the frontend depends on, so the model behind it can change freely.
- Endpoints in `main.py` list engines, serve a per-cycle **snapshot**, return
  **history**, accept an **ad-hoc prediction** (POST), and **simulate** the
  future.
- Heavy model work runs **once at startup** (`service.py`) and is cached, so
  requests are instant.

Next: **file 8**, the dashboard that consumes these endpoints and paints them
onto a 3D engine.
