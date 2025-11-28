# Redis Down Runbook

> **Doc Status (2025-11-27): Active Runbook**  
> **Role:** Operational guide for diagnosing and restoring Redis when the `RedisDown` alert fires.
>
> **SSoT alignment:** Derived over the **Operational SSoT** for deployment and caching:
>
> - Deployment and topology configs (`docker-compose*.yml`, `src/server/config/**/*.ts`).
> - Caching and rate limiting behaviour as implemented in `src/server/cache/redis.ts` and `src/server/middleware/rateLimiter.ts`.
>
> **Precedence:** Actual Redis configuration, connection strings, and server code are authoritative. On conflict, **code + configs win**.

---

## 1. When This Alert Fires

**Alert:** `RedisDown`  
**Condition:** `ringrift_service_status{service="redis"} == 0` for more than 1 minute.

**Impact:**

- Rate limiting and some caching are degraded or disabled.
- Depending on deployment, login/session behaviour may be affected.

---

## 2. Triage

1. Confirm alert in Alertmanager and Grafana.
2. Check container and logs:
   ```bash
   docker compose ps redis
   docker compose logs -f redis
   ```
3. Test connectivity from the app container:
   ```bash
   docker compose exec app node -e "const Redis = require('ioredis'); const r = new Redis(process.env.REDIS_URL); r.ping().then(console.log).catch(console.error);"
   ```

---

## 3. Remediation

1. Restart Redis if process is unhealthy:
   ```bash
   cd /opt/ringrift
   docker compose restart redis
   ```
2. If using a managed Redis service, follow provider guidance (console restart, failover, etc.).
3. Once Redis is healthy, verify application readiness and rate limiting recovery.

---

## 4. Validation

- [ ] `redis-cli PING` returns `PONG` from within the cluster.
- [ ] Application readiness reports Redis as healthy.
- [ ] Rate limiting metrics return to expected baseline.

---

## 5. TODO / Environment-Specific Notes

- Document managed Redis URLs, authentication, and TLS requirements.
- Add links to any internal dashboards specific to Redis performance.
