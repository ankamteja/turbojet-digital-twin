// api.js — thin fetch wrappers around the digital-twin backend.
// The backend already normalises everything into the EngineState shape,
// so these helpers just do the HTTP round-trip and surface errors.

// The backend runs as a separate service, so we can't use location.origin.
// Local dev: uvicorn on :8000. Deployed: the build step replaces the
// __BACKEND_URL__ placeholder with the backend service's real host, so the
// same file works in both places without editing.
const LOCAL = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const BASE = LOCAL ? 'http://localhost:8000' : 'https://__BACKEND_URL__';

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
