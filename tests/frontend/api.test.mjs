// api.test.mjs — the cold-start retry behaviour in getEngines().
//
// api.js reads `location` and calls `fetch` as browser globals. Rather than
// pull in jsdom for two globals, this stubs just those two per test.

import assert from 'node:assert/strict';
import { test } from 'node:test';

function stubBrowserGlobals({ hostname = 'turbojet-dashboard.onrender.com' } = {}) {
  global.location = { hostname };
}

async function loadApi() {
  // Fresh module per test — BASE is computed once at import time from
  // `location.hostname`, so a cached import would keep the first test's URL.
  return import(`../../src/frontend/js/api.js?t=${Date.now()}-${Math.random()}`);
}

test('getEngines returns immediately when the backend answers first try', async () => {
  stubBrowserGlobals();
  let calls = 0;
  global.fetch = async () => {
    calls++;
    return { ok: true, json: async () => [{ engine_id: 1, max_cycle: 30 }] };
  };

  const { getEngines } = await loadApi();
  const result = await getEngines({ baseDelayMs: 1 });

  assert.equal(calls, 1);
  assert.deepEqual(result, [{ engine_id: 1, max_cycle: 30 }]);
});

test('getEngines retries through a cold start and eventually succeeds', async () => {
  stubBrowserGlobals();
  let calls = 0;
  global.fetch = async () => {
    calls++;
    if (calls < 3) {
      // simulates the backend container still coming up
      return { ok: false, status: 502, statusText: 'Bad Gateway' };
    }
    return { ok: true, json: async () => [{ engine_id: 1, max_cycle: 30 }] };
  };

  const retries = [];
  const { getEngines } = await loadApi();
  const result = await getEngines({
    baseDelayMs: 1,
    onRetry: (attempt, total) => retries.push([attempt, total]),
  });

  assert.equal(calls, 3, 'should have failed twice before the third call succeeded');
  assert.deepEqual(retries, [[1, 6], [2, 6]]);
  assert.deepEqual(result, [{ engine_id: 1, max_cycle: 30 }]);
});

test('getEngines gives up and throws after exhausting all attempts', async () => {
  stubBrowserGlobals();
  global.fetch = async () => ({ ok: false, status: 503, statusText: 'Service Unavailable' });

  const { getEngines } = await loadApi();
  await assert.rejects(
    () => getEngines({ attempts: 3, baseDelayMs: 1 }),
    /503/,
    'should surface the last failure, not swallow it',
  );
});

test('per-cycle fetches (getState) do not retry', async () => {
  // Once the backend is up, a stage-scrub failing should fail fast rather
  // than silently retrying for tens of seconds while the UI looks frozen.
  stubBrowserGlobals();
  let calls = 0;
  global.fetch = async () => {
    calls++;
    return { ok: false, status: 500, statusText: 'Internal Server Error' };
  };

  const { getState } = await loadApi();
  await assert.rejects(() => getState(1, 10));
  assert.equal(calls, 1);
});
