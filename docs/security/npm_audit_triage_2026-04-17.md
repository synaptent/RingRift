# npm Audit Triage (2026-04-17)

Read-only analysis of `npm audit` output for the root `package.json` /
`package-lock.json` in the RingRift repository. No dependencies were installed,
upgraded, or otherwise modified by this triage; this document only records
findings and proposed remediations for review.

Inputs:

- Raw audit JSON: `/tmp/ringrift_npm_audit.json`
- Human summary: `/tmp/ringrift_npm_audit_summary.txt`
- Lockfile: `/Users/armand/Development/RingRift/package-lock.json`
- Manifest: `/Users/armand/Development/RingRift/package.json`

---

## Summary

- **17 vulnerable packages**, **19 underlying advisories**
  (0 critical, 7 high, 6 moderate, 4 low — matches TODO.md "Final Quality Sweep").
- **Top root-cause direct deps (fixing these fixes most transitive findings):**
  1. `vite-plugin-pwa@^1.2.0` (devDep) — root cause for
     `workbox-build`, `@rollup/plugin-terser`, `serialize-javascript`,
     `lodash` (6 advisories, 4 HIGH + 2 MODERATE).
  2. `vite@^7.3.1` (devDep, direct) — 3 advisories (2 HIGH + 1 MODERATE).
  3. `prisma@^7.4.2` / `@prisma/client@^7.4.2` — root cause for
     `@prisma/dev` → `@hono/node-server` → `hono` chain (7 MODERATE).
  4. `axios@^1.13.5` (direct prod dep) — 2 MODERATE (SSRF class).
     Also transitively fixes `follow-redirects` MODERATE.
  5. `jest-environment-jsdom@^29.7.0` (devDep) — 1 LOW (`@tootallnate/once`).
- **Zero critical.** Every HIGH advisory is in either a **build-time-only
  devDep** (`vite`, `workbox-build`/`serialize-javascript`, `lodash`, `defu`)
  or requires an attacker-controlled code path that RingRift's production
  runtime does not expose.
- **Runtime / production surface affected:** `axios` (used by both
  `ai-service` HTTP clients and server-side). Everything else is dev/build
  tooling or a devDep Vite/Prisma/Jest chain.
- `npm audit fix` (non-forced) is reported safe for the Vite/axios/lodash/
  defu/follow-redirects/hono/serialize-javascript/workbox-build chains.
- The `prisma`, `vite-plugin-pwa`, and `jest-environment-jsdom` fixes are
  flagged **semver-major** by npm and must be handled manually — in the
  Prisma and vite-plugin-pwa cases the "fix" is a **downgrade** (Prisma
  6.19.3, vite-plugin-pwa 0.19.8) which we do **not** recommend; prefer
  overrides instead.

---

## Recommended upgrade plan (ordered by impact × ease)

| #   | Action                                                                                   | Fixes                                                                                                           | Risk                                                                                                                     |
| --- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | Upgrade direct `axios` to `^1.15.0` (safe `npm audit fix`)                               | GHSA-3p68-rc4w-qgx5, GHSA-fvcv-3m26-pcqx, GHSA-r4q5-vmmm-2653 (3 MODERATE, incl. transitive `follow-redirects`) | Low — axios 1.x patch-level bump; we already use `^1.13.5`.                                                              |
| 2   | Upgrade direct `vite` to latest `^7.3.2+` (safe `npm audit fix`)                         | GHSA-4w7w-66w2-5vf9, GHSA-v2wj-q39q-566r, GHSA-p9ff-h696-f583 (2 HIGH + 1 MODERATE)                             | Low — dev-server only; SemVer-minor.                                                                                     |
| 3   | Add `npm overrides` for `serialize-javascript` (`>=7.0.5`) and `lodash` (`>=4.17.24`)    | GHSA-5c6j-r48x-rmvq (HIGH), GHSA-qj8w-gfj5-8c6v, GHSA-r5fr-rjxr-66jc (HIGH), GHSA-f23m-r3pf-42rh                | Low — both are internal tooling deps of `workbox-build`; avoids the recommended _downgrade_ to `vite-plugin-pwa@0.19.8`. |
| 4   | Add `npm override` for `defu` (`>=6.1.5`)                                                | GHSA-737v-mqg7-c878 (HIGH, prototype pollution)                                                                 | Low — `defu` is only reached via Prisma devtooling; override is preferable to Prisma major bump.                         |
| 5   | Add `npm overrides` for `hono` (`>=4.12.14`) and `@hono/node-server` (`>=1.19.13`)       | All 7 `hono` / `@hono/node-server` MODERATE (6 hono + 1 @hono/node-server)                                      | Low — reached only through `prisma` devtools; we do **not** ship Hono in production.                                     |
| 6   | Plan a separate `jest-environment-jsdom` → `^30.3.0` upgrade for the Jest 30 migration   | GHSA-vpq2-c234-7xj6 (LOW)                                                                                       | Medium — Jest 30 SemVer-major; defer until after the Final Quality Sweep.                                                |
| 7   | Fix Husky `prepare` script (change `husky install` → `husky`, remove deprecated command) | Husky v10 deprecation warning                                                                                   | Low — cosmetic.                                                                                                          |

Item 1 and 2 can be done today with `npm audit fix` (scoped, non-forced).
Items 3–5 must be handled with manual `overrides` to avoid the
SemVer-major recommendations returned by `npm audit`. Item 6 can be
scheduled in the next Jest maintenance window.

### Suggested `package.json` `overrides` snippet

> NOTE: For review only — not applied by this triage. Verify each version
> exists on the npm registry at application time.

```jsonc
{
  "overrides": {
    "serialize-javascript": ">=7.0.5",
    "lodash": ">=4.17.24",
    "defu": ">=6.1.5",
    "hono": ">=4.12.14",
    "@hono/node-server": ">=1.19.13",
  },
}
```

After editing `overrides`, regenerate the lockfile with `npm install`
(not `npm audit fix --force`) and re-run `npm audit` to confirm.

---

## Detailed findings

Indirection chains were resolved from `package-lock.json`.

### GHSA-4w7w-66w2-5vf9 (MODERATE) — `vite`

- **Role:** devDep, **direct** (`vite@^7.3.1`).
- **Range vulnerable:** `>=7.0.0 <=7.3.1`. CWEs: 22, 200 (path traversal,
  info disclosure in optimized-deps `.map` handling).
- **Fix:** `npm audit fix` (upgrade `vite` patch-level to the first `7.3.x`
  release above `7.3.1`).
- **Production impact:** None. Vite dev server is not exposed in
  ringrift.ai production (the server uses `dist/server` and `vite build`
  static output).

### GHSA-v2wj-q39q-566r (HIGH) — `vite`

- CWEs 180, 284. `server.fs.deny` bypass via query string.
- Range `>=7.1.0 <=7.3.1`. Same fix as above.
- Production impact: None (same rationale — dev-server only).

### GHSA-p9ff-h696-f583 (HIGH) — `vite`

- Arbitrary file read via WebSocket in dev server. CWEs 200, 306.
- Range `>=7.0.0 <=7.3.1`.
- Recommendation: **Upgrade direct `vite` to `^7.3.2+`** (item 2).

### GHSA-3p68-rc4w-qgx5 (MODERATE) — `axios`

- **Role:** prod dep, **direct** (`axios@^1.13.5`).
- NO_PROXY hostname normalization bypass → SSRF. CVSS 4.8.
- Range vulnerable: `>=1.0.0 <1.15.0`.
- **Fix:** upgrade `axios` to `^1.15.0` (non-major).
- **Production impact:** Potentially reachable from any server-side HTTP
  call using `axios` with user-controlled URLs. Review: most server axios
  usage is internal (`AI_SERVICE_URL`, Sentry, AWS SES via SDK). No known
  attacker-controlled URL path, but this is the one finding that warrants
  a **ringrift.ai production redeploy** once upgraded.

### GHSA-fvcv-3m26-pcqx (MODERATE) — `axios`

- Cloud metadata exfiltration via header injection chain. CWEs 113, 444, 918.
- Range vulnerable: `>=1.0.0 <1.15.0`.
- Same fix as above. Same production-impact note.

### GHSA-r4q5-vmmm-2653 (MODERATE) — `follow-redirects`

- Auth header leak on cross-domain redirect.
- **Role:** transitive under `axios` (`follow-redirects@1.15.11`; vulnerable range `<=1.15.11`).
- **Fix:** Bumping `axios` to `^1.15.0` pulls `follow-redirects@^1.15.12+`
  (verify on install). No separate override needed.

### GHSA-5c6j-r48x-rmvq (HIGH) — `serialize-javascript`

- RCE via `RegExp.flags` and `Date.prototype.toISOString()`. CWE-96. CVSS 8.1.
- Range vulnerable: `<=7.0.2`.
- **Role:** transitive under `@rollup/plugin-terser` ← `workbox-build` ←
  `vite-plugin-pwa`.
- **Fix (recommended):** `npm overrides` → `serialize-javascript: ">=7.0.5"`.
  - npm's auto-fix would downgrade `vite-plugin-pwa` from 1.2.0 → 0.19.8
    (SemVer-major **downgrade**) which removes features; overriding the
    transitive is strictly better.
- **Production impact:** Build-time only — `workbox-build` runs inside
  `vite build`. An attacker able to inject malicious data into the PWA
  build input could achieve RCE on the build host (CI). Not a live
  ringrift.ai prod runtime risk, but important for CI/CD hygiene.

### GHSA-qj8w-gfj5-8c6v (MODERATE) — `serialize-javascript`

- CPU DoS via crafted array-like objects. Range `<7.0.5`.
- Same chain and fix as above.

### GHSA-r5fr-rjxr-66jc (HIGH) — `lodash`

- Code injection via `_.template` imports-key names. CWE-94. CVSS 8.1.
- Range vulnerable: `>=4.0.0 <=4.17.23` (installed: 4.17.23).
- **Role:** transitive under `workbox-build@>=7.1.0` ← `vite-plugin-pwa`.
- **Fix (recommended):** `npm overrides` → `lodash: ">=4.17.24"`.
- **Production impact:** Build-time only (same rationale as
  serialize-javascript). `_.template` is not used at runtime by RingRift
  TS/React code (the `lodash.*` dot-subpackages in the lockfile are a
  separate, unaffected subtree).

### GHSA-f23m-r3pf-42rh (MODERATE) — `lodash`

- Prototype pollution in `_.unset`/`_.omit`. Same range, same fix.

### GHSA-737v-mqg7-c878 (HIGH) — `defu`

- Prototype pollution via `__proto__` key in defaults. CVSS 7.5.
- Range `<=6.1.4` (installed: 6.1.4).
- **Role:** transitive under `c12` / `giget` / `rc9` ← `@prisma/dev` ← `prisma`
  devtools.
- **Fix (recommended):** `npm overrides` → `defu: ">=6.1.5"`.
  - npm's auto-fix would do a `prisma` major **downgrade** (6.19.3); we
    prefer overriding the transitive.
- **Production impact:** None in production runtime. `@prisma/dev` is a
  local devtool (`prisma dev`) and is never shipped. Build image excludes
  Prisma dev devDeps.

### GHSA-92pp-h63x-v22m (MODERATE) — `@hono/node-server`

- Middleware bypass via repeated slashes in `serveStatic`.
- Range `<1.19.13`. **Role:** transitive under `@prisma/dev`.
- **Fix (recommended):** override `@hono/node-server: ">=1.19.13"`.

### GHSA-26pp-8wgv-hjvm, GHSA-r5rp-j6wh-rvv4, GHSA-xpcf-pg52-r92g, GHSA-xf4j-xp2r-rqqx, GHSA-wmmm-f939-6g9c, GHSA-458j-xx4x-4375 (MODERATE ×6) — `hono`

- Cookie validation, IP-mapping, path traversal in `toSSG`, middleware
  bypass, JSX-SSR HTML injection. All MODERATE, all fixed in `4.12.14+`.
- Range installed: `<=4.12.13` under `@prisma/dev → @hono/node-server → hono`.
- **Fix (recommended):** override `hono: ">=4.12.14"`.
- **Production impact:** None — Hono is never reachable from RingRift
  server/client code at runtime. `@prisma/dev` is devtool-only.

### GHSA-vpq2-c234-7xj6 (LOW) — `@tootallnate/once`

- Incorrect control flow scoping. CVSS 3.3. Range `<3.0.1`.
- **Role:** transitive `http-proxy-agent` ← `jsdom` ← `jest-environment-jsdom@29.7.0`.
- **Fix:** schedule a Jest 30 upgrade — `jest-environment-jsdom@^30.3.0`.
  In the meantime, low severity + test-only → accept risk.
- **Production impact:** None (test-only).

---

## Production deploy impact

| Fix                                                                                 | ringrift.ai redeploy needed?                                          | Why                                                                                                       |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| axios `^1.15.0`                                                                     | **Yes** — restart Node server (`pm2 restart ringrift`) after rebuild. | `axios` is loaded by the production `dist/server/` bundle.                                                |
| vite `^7.3.2+`                                                                      | No                                                                    | Dev-server only; not in production bundle. Client static assets should still be rebuilt via CI.           |
| `serialize-javascript` / `lodash` / `defu` / `hono` / `@hono/node-server` overrides | No (runtime); yes for **CI image**                                    | Transitive devDeps only; however, CI/build hosts should regenerate `node_modules` with the new overrides. |
| `jest-environment-jsdom` 30.x                                                       | No                                                                    | Test-only.                                                                                                |
| Husky script change                                                                 | No                                                                    | Developer-only `prepare` hook.                                                                            |

**Net:** only the `axios` upgrade warrants a production deploy of the
ringrift.ai Node server (PM2 restart). All other fixes are dev/build-time
hygiene.

---

## Husky deprecation

**Symptom (reproduced during `npm install`):**

```
husky - install command is DEPRECATED
```

Current manifest:

```jsonc
"prepare": "husky install"
```

**Background:** Husky ≥ 9 replaced `husky install` with `husky` (no
subcommand). In Husky v10 the old subcommand will be removed entirely.
RingRift is on `husky@^9.1.7`, so the fix is cosmetic today but will
start failing on the first Husky v10 install.

**Safe fix (no code change beyond `package.json`):**

```jsonc
"scripts": {
  ...
  "prepare": "husky",
  ...
}
```

No need to uninstall Husky — hooks in `.husky/` continue to work unchanged
with the new `prepare` command. Keep the existing `.husky/pre-commit` et
al.; Husky v9 will continue to migrate legacy hook shims automatically.

**Do not** remove Husky entirely — it is still wired into
`lint-staged` and the repo's pre-commit quality gate. Removing Husky
would silently disable per-commit lint/typecheck hooks.

---

## What was explicitly _not_ done

- No `npm audit fix` was run.
- No `npm audit fix --force` was run (TODO.md explicitly forbids it).
- No `package.json`, `package-lock.json`, or `.husky/` edits were made.
- No `git add` / `git commit` / `git push`.

The plan above is for reviewer approval; once approved, a follow-up
change can:

1. Edit `package.json` to add `overrides` + the Husky `prepare` fix.
2. Run `npm install` (not `audit fix --force`) to regenerate the lockfile.
3. Re-run `npm audit` and attach the residual output (expect: 0
   high/critical, at most 1 LOW `@tootallnate/once` pending the Jest-30
   migration).
4. Run `npm run build && npm test` to confirm no regressions.
5. Redeploy `ringrift.ai` via PM2 (only required for the `axios` bump).

---

## Reviewer

Droid (Claude Opus 4.7) via factory.ai
