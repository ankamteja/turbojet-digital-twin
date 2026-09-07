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

// Render's free tier sleeps the backend after ~15 min idle; a visitor waking
// it eats a ~30-60s cold start where requests fail (a timed-out fetch, or a
// 502/503 while the container is still coming up) before the service is
// actually ready. A single failed getJSON on boot used to read as "backend
// unreachable" even though it just needed more time. Retries with backoff so
// the same wake-up window that only costs a human "hit refresh" reads as a
// normal (if slow) first load instead of an error.
//
// Not used for interactive per-cycle fetches — those fail fast on purpose so
// scrubbing feels responsive once the backend is actually up.
async function getJSONWithRetry(path, { attempts = 6, baseDelayMs = 2000, onRetry } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await getJSON(path);
    } catch (err) {
      lastErr = err;
      if (i === attempts - 1) break;
      if (onRetry) onRetry(i + 1, attempts);
      // linear backoff is enough here: we're waiting out a fixed-length cold
      // start, not backing off from a server we're overloading.
      await new Promise((r) => setTimeout(r, baseDelayMs * (i + 1)));
    }
  }
  throw lastErr;
}

// GET /api/engines -> [{engine_id, max_cycle}, ...]
// Called once on boot, so it's the request that pays for a cold start.
export function getEngines(opts) {
  return getJSONWithRetry('/api/engines', opts);
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
