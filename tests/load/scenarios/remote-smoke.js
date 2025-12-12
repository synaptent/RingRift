/**
 * RingRift Remote Smoke Load Test
 *
 * Lightweight, safe-to-run scenario intended for production/staging endpoints.
 * Uses small VU/duration defaults and the shared auth + API helpers.
 *
 * Usage (example):
 *   k6 run --insecure-skip-tls-verify \
 *     --env BASE_URL=https://<public-host-or-ip> \
 *     --env LOADTEST_EMAIL=loadtest@test.local \
 *     --env LOADTEST_PASSWORD=LoadTest123 \
 *     tests/load/scenarios/remote-smoke.js
 */

import { check, sleep } from 'k6';
import { loginAndGetToken } from '../auth/helpers.js';
import { createGame } from '../helpers/api.js';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '1m',
  thresholds: {
    checks: ['rate>0.95'],
  },
  tags: {
    scenario: 'remote-smoke',
    test_type: 'smoke',
    environment: __ENV.THRESHOLD_ENV || 'production',
  },
};

export function setup() {
  const { token } = loginAndGetToken(BASE_URL, {
    tags: { name: 'auth-login-setup' },
  });
  return { token };
}

export default function (data) {
  const { res, success } = createGame(data.token, {
    boardType: __ENV.BOARD_TYPE || 'square8',
    maxPlayers: Number(__ENV.MAX_PLAYERS || 2),
    isPrivate: true,
    isRated: false,
    tags: { name: 'create-game' },
  });

  check(res, {
    'create-game status 201': (r) => r.status === 201,
    'create-game success envelope': () => success,
  });

  sleep(1);
}

