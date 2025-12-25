# Security Incidents

This guide covers security-related incidents including rate limiting alerts, suspicious activity, and security breach response.

## Alerts Covered

| Alert                 | Severity | Threshold            | Duration |
| --------------------- | -------- | -------------------- | -------- |
| HighRateLimitHits     | Warning  | > 1/sec per endpoint | 5 min    |
| SustainedRateLimiting | Warning  | > 10/sec total       | 15 min   |

**Note**: Security incidents may also manifest through other alerts (high error rates, unusual traffic patterns, etc.)

---

## Alert: HighRateLimitHits

### Severity

**P3 Medium** - Rate limiting being triggered frequently on specific endpoint

### Symptoms

- Users receiving 429 (Too Many Requests) responses
- Rate limit counters increasing rapidly
- Specific endpoint under heavy load
- `sum(rate(ringrift_rate_limit_hits_total[5m])) by (endpoint) > 1`

### Impact

- Legitimate users may be blocked
- Potential abuse attempt
- Server resources being consumed

### Initial Triage (5 min)

```bash
# 1. Check which endpoints are affected
curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total

# 2. Check current request rate by endpoint
docker compose logs --tail 500 app | grep "429" | \
  awk '{print $NF}' | sort | uniq -c | sort -rn

# 3. Check for specific IP patterns
docker compose logs --tail 1000 app | grep "429" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c | sort -rn

# 4. Check overall traffic
curl -s http://localhost:3000/metrics | grep http_requests_total
```

### Diagnosis

#### Identify the Source

```bash
# Check rate limited IPs
docker compose logs --tail 2000 app | grep "rate.limit\|429" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c | sort -rn | head -20

# Check rate limited users (if logged)
docker compose logs --tail 2000 app | grep "rate.limit\|429" | \
  grep -oP 'user[=:]\K[^ ]+' | sort | uniq -c | sort -rn | head -20

# Check User-Agent patterns
docker compose logs --tail 2000 app | grep "429" | \
  grep -oP 'user-agent[=:]\K[^"]+' | sort | uniq -c | sort -rn | head -10
```

#### Assess Intent

| Pattern                       | Likely Cause       | Response                     |
| ----------------------------- | ------------------ | ---------------------------- |
| Single IP, single endpoint    | API abuse/scraping | Consider IP block            |
| Single IP, multiple endpoints | Bot/automated tool | Consider IP block            |
| Multiple IPs, same pattern    | Distributed attack | Investigate source           |
| Legitimate user agents        | Traffic spike      | May need to adjust limits    |
| Bot/script user agents        | Automated abuse    | Block user agent or IP range |

### Mitigation

#### Monitor and Assess

If rate limiting is working as expected and protecting the service:

```bash
# Monitor rate limit effectiveness
watch -n 10 'curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total'

# Ensure service is healthy despite rate limits
curl -s http://localhost:3000/health | jq
```

#### Block Specific IP (if abuse confirmed)

```bash
# Using nginx (if fronting the app)
# Add to nginx.conf deny list
echo "deny 1.2.3.4;" >> /etc/nginx/conf.d/block.conf
nginx -s reload

# Or using iptables (host-level)
sudo iptables -A INPUT -s 1.2.3.4 -j DROP
```

#### Adjust Rate Limits (if legitimate traffic)

```bash
# If rate limits are too aggressive for legitimate use
# Edit rate limit configuration (check app config)
# May need to adjust RATE_LIMIT_* environment variables

# After adjustment, restart app
docker compose restart app
```

### Communication

- **Slack**: Post in #security (if pattern looks malicious)
- **Slack**: Post in #alerts (if traffic spike)
- **Document**: Log IPs/patterns for future reference

---

## Alert: SustainedRateLimiting

### Severity

**P3 Medium** - Persistent rate limiting across multiple endpoints

### Symptoms

- Rate limiting happening continuously
- Multiple endpoints affected
- Sustained over 15+ minutes

### Impact

- Multiple users potentially affected
- Possible coordinated abuse
- Service under sustained load

### Diagnosis

```bash
# Check total rate limit hits across all endpoints
curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total | \
  awk '{sum += $2} END {print "Total:", sum}'

# Check breakdown by endpoint
curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total

# Analyze traffic patterns
docker compose logs --since 15m app | grep -c "429"
docker compose logs --since 15m app | grep -c "200"
```

### Assessment

**Legitimate traffic spike:**

- Occurs during peak hours
- Distributed sources
- Normal user agents

**Potentially malicious:**

- Unusual hours
- Concentrated sources
- Automated patterns
- Targeting specific endpoints

### Mitigation

For sustained rate limiting, consider:

1. **If legitimate traffic**: Temporarily increase rate limits
2. **If abuse**: Implement IP blocking or stricter rules
3. **If attack**: Activate DDoS protection if available

---

## Security Incident Response

### Types of Security Incidents

| Type               | Indicators                                 | Severity    |
| ------------------ | ------------------------------------------ | ----------- |
| DDoS Attempt       | Massive traffic spike, service degradation | P1 Critical |
| Brute Force        | High rate of failed logins                 | P2 High     |
| API Abuse          | Rate limiting + unusual patterns           | P3 Medium   |
| Data Exfiltration  | Unusual data access patterns               | P1 Critical |
| Account Compromise | Suspicious account activity                | P2 High     |

---

## Incident: DDoS Attempt

### Indicators

- Sudden massive traffic increase
- Rate limiting at maximum
- Service becoming unresponsive
- Traffic from unusual sources

### Immediate Actions

```bash
# 1. Confirm it's a DDoS (not legitimate traffic)
curl -s http://localhost:3000/metrics | grep http_requests_total

# 2. Check service health
curl -s http://localhost:3000/health | jq

# 3. Identify attacking IPs
docker compose logs --tail 5000 app | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | \
  sort | uniq -c | sort -rn | head -50
```

### Mitigation

```bash
# Enable DDoS mitigation (if available)
# Cloud providers often have DDoS protection

# Block top attacking IPs
TOP_IPS=$(docker compose logs --tail 5000 app | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | \
  sort | uniq -c | sort -rn | head -20 | awk '{print $2}')

for ip in $TOP_IPS; do
  sudo iptables -A INPUT -s $ip -j DROP
done

# Reduce rate limits temporarily
# Lower the threshold to block more aggressively
```

### Communication

- **Status Page**: "Service experiencing high traffic - Working to restore"
- **Slack**: Escalate to #incidents and #security
- **Escalation**: Involve infrastructure team

---

## Incident: Brute Force Attack

### Indicators

- High rate of failed login attempts
- Targeting login/auth endpoints
- Multiple usernames or single username

### Detection

```bash
# Check for failed auth attempts
docker compose logs --tail 2000 app | grep -E "auth.*fail|login.*fail|401" | wc -l

# Check targeting pattern
docker compose logs --tail 2000 app | grep -E "auth.*fail|login.*fail" | \
  grep -oP 'user[=:]\K[^ ]+' | sort | uniq -c | sort -rn | head -20

# Check source IPs
docker compose logs --tail 2000 app | grep -E "auth.*fail|login.*fail" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c | sort -rn | head -20
```

### Mitigation

```bash
# Block attacking IPs
# (Use appropriate method for your infrastructure)

# If targeting specific account, consider locking it temporarily
# This requires database access
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "UPDATE users SET locked_until = now() + interval '1 hour' WHERE username = 'targeted_user';"

# Increase login rate limiting
# Edit rate limit configuration for auth endpoints
```

### Post-Incident

- Review affected accounts
- Notify users if accounts were at risk
- Review auth security (2FA, password policy)

---

## Incident: Data Exfiltration Suspicion

### Indicators

- Unusual bulk data requests
- Single user accessing large amounts of data
- API endpoints being crawled systematically

### Detection

```bash
# Check for bulk data access patterns
docker compose logs --tail 5000 app | grep -E "GET.*/api" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c | sort -rn | head -20

# Check for systematic crawling
docker compose logs --tail 5000 app | grep "GET" | \
  awk '{print $NF}' | sort | uniq -c | sort -rn | head -30
```

### Immediate Actions

1. **Identify the actor** (IP, user account)
2. **Block access** immediately
3. **Preserve logs** for investigation
4. **Assess what was accessed**

```bash
# Block the IP immediately
sudo iptables -A INPUT -s <suspicious_ip> -j DROP

# Preserve logs
docker compose logs --since 24h app > /tmp/incident-logs-$(date +%Y%m%d).log

# If user account involved, disable it
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "UPDATE users SET disabled = true WHERE id = '<user_id>';"
```

### Communication

- **Slack**: Immediately escalate to #security
- **Management**: Inform leadership
- **Legal**: May need to notify depending on data accessed

---

## Incident: Account Compromise

### Indicators

- User reports unauthorized access
- Unusual activity from user account
- Multiple sessions from different locations

### Investigation

```bash
# Check recent activity for account
docker compose logs --tail 10000 app | grep "user_id_here"

# Check session patterns
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "SELECT * FROM sessions WHERE user_id = '<user_id>' ORDER BY created_at DESC LIMIT 20;"

# Check login locations/IPs
docker compose logs --tail 10000 app | grep "login.*user_id" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq
```

### Immediate Actions

```bash
# 1. Invalidate all sessions for user
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "DELETE FROM sessions WHERE user_id = '<user_id>';"

# 2. Force password reset (if feature exists)
# Or disable account temporarily
docker exec ringrift-postgres-1 psql -U ringrift -c \
  "UPDATE users SET requires_password_reset = true WHERE id = '<user_id>';"

# 3. Document the timeline
```

### Communication

- Contact the affected user
- Document the incident
- Review for wider compromise

---

## Security Monitoring Commands

### Traffic Analysis

```bash
# Request rate by IP
docker compose logs --tail 10000 app | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | \
  sort | uniq -c | sort -rn | head -30

# Request rate by endpoint
docker compose logs --tail 10000 app | \
  grep -oP 'GET|POST|PUT|DELETE \K[^ ]+' | \
  sort | uniq -c | sort -rn | head -30

# Status code distribution
docker compose logs --tail 10000 app | \
  grep -oP 'status[=:]\K[0-9]+' | \
  sort | uniq -c | sort -rn
```

### Authentication Analysis

```bash
# Failed logins
docker compose logs --tail 5000 app | grep -c "401"

# Successful logins
docker compose logs --tail 5000 app | grep "login.*success\|auth.*success" | wc -l

# Auth attempts by IP
docker compose logs --tail 5000 app | grep -E "login|auth" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | \
  sort | uniq -c | sort -rn | head -20
```

### Rate Limit Status

```bash
# Current rate limit hits
curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total

# Rate limit by endpoint
curl -s http://localhost:3000/metrics | grep ringrift_rate_limit_hits_total | \
  awk -F'[{}"]' '{print $3, $NF}'
```

---

## IP Blocking Reference

### Temporary Block (iptables)

```bash
# Block single IP
sudo iptables -A INPUT -s 1.2.3.4 -j DROP

# Block IP range
sudo iptables -A INPUT -s 1.2.3.0/24 -j DROP

# List blocks
sudo iptables -L INPUT -n

# Remove block
sudo iptables -D INPUT -s 1.2.3.4 -j DROP
```

### Persistent Block (nginx)

```nginx
# Add to nginx configuration
location / {
    deny 1.2.3.4;
    deny 5.6.7.0/24;
    # ... rest of config
}
```

### Check if IP is already blocked

```bash
sudo iptables -L INPUT -n | grep 1.2.3.4
```

---

## Escalation

### When to Escalate Immediately

| Situation                       | Escalate To                |
| ------------------------------- | -------------------------- |
| Active data exfiltration        | Security Lead + Management |
| Confirmed breach                | Security Lead + Legal      |
| DDoS severely impacting service | Infrastructure + Security  |
| Credential compromise (admin)   | Security Lead + Management |

### Security Contact List

| Role             | Contact     | Purpose                |
| ---------------- | ----------- | ---------------------- |
| Security Lead    | [Configure] | All security incidents |
| Infrastructure   | [Configure] | DDoS, network issues   |
| Legal/Compliance | [Configure] | Data breach, GDPR      |
| Management       | [Configure] | Significant incidents  |

---

## Post-Incident Checklist

After any security incident:

- [ ] Incident documented (timeline, actions, outcome)
- [ ] Logs preserved
- [ ] Root cause identified
- [ ] Vulnerabilities patched (if applicable)
- [ ] Monitoring enhanced (if gap found)
- [ ] Affected users notified (if applicable)
- [ ] Post-mortem scheduled
- [ ] Security review scheduled (if significant)

---

## Related Documentation

- [Initial Triage](TRIAGE_GUIDE.md)
- [Availability Incidents](AVAILABILITY.md)
- [Security Threat Model](../security/SECURITY_THREAT_MODEL.md)
- [Secrets Management](../operations/SECRETS_MANAGEMENT.md)
- [Post-Mortem Template](POST_MORTEM_TEMPLATE.md)
