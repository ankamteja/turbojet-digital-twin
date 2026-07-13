// api.js — thin fetch wrappers around the digital-twin backend.
// The backend already normalises everything into the EngineState shape,
// so these helpers just do the HTTP round-trip and surface errors.

// Backend runs separately (uvicorn on :8000). We hard-code the origin rather
// than deriving it from location.origin because the frontend is served from a
// different port (python -m http.server), so same-origin would break.
const BASE = 'http://localhost:8000';

async function getJSON(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    throw new Error(`API ${path} -> ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// GET /api/engines -> [{engine_id, max_cycle}, ...]
export function getEngines() {
  return getJSON('/api/engines');
}

// GET /api/engine/{id}/state?cycle=n -> EngineState
export function getState(id, cycle) {
  return getJSON(`/api/engine/${id}/state?cycle=${cycle}`);
}

// GET /api/engine/{id}/history -> per-cycle predictions + ground truth
export function getHistory(id) {
  return getJSON(`/api/engine/${id}/history`);
}

// GET /api/engine/{id}/simulate?to_cycle=m -> projected future health
export function getSimulate(id, toCycle) {
  return getJSON(`/api/engine/${id}/simulate?to_cycle=${toCycle}`);
}
