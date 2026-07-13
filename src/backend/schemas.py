"""API contract — the JSON shapes the frontend depends on.

The frontend only ever sees these models. The model internals (features,
ensemble size, losses) can change freely as long as EngineState stays stable.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Conditions(BaseModel):
    altitude_m: float
    mach: float
    tamb_k: float
    pamb_pa: float
    rpm: float
    fuel_flow_kg_s: float


class ComponentHealth(BaseModel):
    value: float                       # 0–1 health index
    confidence: float                  # 0–1 (ensemble agreement)
    trend: float                       # per-cycle slope (negative = degrading)
    rul_cycles: Optional[int]          # None = no finite life on current trend
    recommendation: str
    contributing: dict[str, float]     # key sensor/ratio values behind the call


class PerfValue(BaseModel):
    value: float
    confidence: float


class Performance(BaseModel):
    thrust_n: PerfValue
    tsfc_g_n_s: PerfValue


class Alert(BaseModel):
    component: str
    level: str                         # "info" | "warn" | "critical"
    msg: str


class HistoryPoint(BaseModel):
    cycle: int
    overall: float
    thrust_n: float


class Sensors(BaseModel):
    """Raw gas-path station readings, passed through for HUD gauges.

    EGT (exhaust/turbine gas temp) = t4_k; station pressures drive the
    pressure-ratio and thermal displays.
    """
    p2_pa: float
    t2_k: float
    p3_pa: float
    t3_k: float
    p4_pa: float
    t4_k: float
    n_pct: float          # spool speed as % of reference max (single-spool N1≈N2)


class EngineState(BaseModel):
    engine_id: int
    cycle: int
    conditions: Conditions
    sensors: Sensors
    health: dict[str, ComponentHealth]  # keys: compressor/combustor/turbine/overall
    performance: Performance
    alerts: list[Alert]
    history: list[HistoryPoint]


class EngineInfo(BaseModel):
    engine_id: int
    max_cycle: int


class SensorRow(BaseModel):
    """Ad-hoc telemetry for POST /api/predict."""
    EngineID: int = 0
    Cycle: int
    Altitude_m: float
    Mach: float
    Tamb_K: float
    Pamb_Pa: float
    RPM_rev_min: float
    FuelFlow_kg_s: float
    P2_Pa: float
    T2_K: float
    P3_Pa: float
    T3_K: float
    P4_Pa: float
    T4_K: float
