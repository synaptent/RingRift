# Production Web Experience Audit

Audit date: April 15, 2026

Scope:

- production host health on `ringrift.ai`
- first-visit browser flow
- local sandbox usability for an anonymous visitor
- obvious browser/runtime errors

This is a findings document only. It does not propose code changes or make any
runtime changes to production.

## Environment Checked

- host: `ubuntu@54.198.219.106`
- app process manager: `pm2`
- reverse proxy: `nginx`
- app health endpoints observed:
  - `http://localhost/health`
  - `http://localhost/api/health`
  - `https://ringrift.ai/health`
- AI health endpoint observed:
  - `http://localhost:8765/health`

## Verified Healthy

- `pm2 list` showed `ringrift-server`, `ringrift-ai`, and `ringrift-ai-dashboard`
  online.
- `curl http://localhost/health` returned `200` with `{"status":"healthy", ...}`.
- `curl http://localhost/api/health` returned `200` with `{"status":"ok", ...}`.
- `curl http://localhost:8765/health` returned `healthy`.
- Anonymous sandbox play still works: a new visitor can launch a sandbox game,
  make a move, and receive AI responses.

## Important Operational Note

The production box is fronted by `nginx` on port `80`. On this host,
`curl localhost:3000/health` was not the right live check and returned
connection-refused during the audit. The real local health path was
`http://localhost/health`, with `nginx` forwarding to the Node app.

That is not a product bug by itself, but it is an operator-footgun because some
older docs still imply direct app-port probing on the host.

## Product Findings

### 1. The root URL drops a cold visitor straight onto the login page

Observed behavior:

- `https://ringrift.ai` redirects to `https://ringrift.ai/login`

Why it matters:

- the login page is functional, but it is not the best first impression for a
  technical stranger deciding in 30 seconds whether the project is interesting
- the strongest part of the product is the game itself, but the project story
  is hidden behind a login-first route

Current mitigation:

- the login page does include a clear `Play Local Sandbox Game` path

Impact:

- high product/onboarding friction

### 2. The sandbox does work for an anonymous visitor

Observed behavior:

- from `/login`, the `Play Local Sandbox Game` CTA opens `/sandbox`
- the sandbox loads without account creation
- a human move was accepted
- the local AI answered and the game continued

Why it matters:

- this is the most important part of the current public experience, and it is
  functioning

Impact:

- positive; this is the current strongest public flow

### 3. The sandbox chooser has visible polish issues

Observed behavior:

- preset labels in the chooser appeared visually jammed together, for example:
  - `Learn the Basicssq8`
  - `Human vs AIsq8`
  - `Hotseatsq8`
  - `Learn the Basicshex8`

Why it matters:

- this is immediately visible on the first playable screen
- it makes the UI feel rougher than the underlying game actually is

Impact:

- medium visual-quality issue

### 4. Browser console shows CSP failures on the first page load

Observed behavior:

- blocked stylesheet from `https://fonts.bunny.net/...`
- blocked inline script on the root/login route

Why it matters:

- first-load console noise makes the app look less production-ready
- blocked fonts can subtly degrade typography and polish

Impact:

- medium frontend-quality issue

### 5. Sandbox play produces repeated `404` requests for `/api/games/sandbox/evaluate`

Observed behavior:

- repeated `404` errors from the browser for `https://ringrift.ai/api/games/sandbox/evaluate`
- the local sandbox still functioned despite these failures

Why it matters:

- this looks like dead or optional analysis wiring still firing in the live UI
- it creates noisy console output and suggests unfinished integration work to
  anyone inspecting the page

Impact:

- medium runtime-noise issue

### 6. Production health is good, but there are two operational watch items

Observed behavior:

- `ringrift-ai` health returned `healthy` but warned on disk usage at `70.8%`
- `pm2` restart counts were non-trivial during the audit window:
  - `ringrift-server`: `44` restarts over `5d`
  - `ringrift-ai`: `22` restarts over `36h`

Why it matters:

- neither issue blocked the product during the audit
- both are signs worth tracking before pushing more public traffic

Impact:

- medium operational concern

## Raw Notes

- `ringrift-server` is behind `nginx`; local host checks should reflect that.
- `ringrift-ai` answered health checks consistently during the audit.
- server logs showed normal `GET /` traffic alongside unrelated hostile bot/CORS
  noise.
- the sandbox is much better than the root-route impression. The product has a
  viable “just try it” path; it is simply not the first thing a visitor sees.

## Recommended Fix Order

This audit does not implement fixes, but the best order is clear:

1. Give the root URL a better visitor path than login-first.
2. Remove the sandbox `404` noise for `/api/games/sandbox/evaluate`.
3. Fix the CSP/font issues.
4. Polish the sandbox chooser card text/layout.
5. Track AI disk usage and restart counts as a separate production-ops follow-up.
