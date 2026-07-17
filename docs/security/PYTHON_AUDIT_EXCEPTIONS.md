# Python Dependency Audit Exceptions

RingRift's Python dependency audit fails on every reported advisory unless a temporary exception
is recorded in [`python_audit_exceptions.json`](python_audit_exceptions.json). Inline
`pip-audit --ignore-vuln` suppressions are not permitted.

The repository wrapper is:

```bash
cd ai-service
python -m pip install 'pip-audit>=2.7.0,<3.0.0'
python scripts/check_python_dependency_audit.py
```

It runs pip-audit in strict JSON mode and validates the report against the ledger. An exception
must identify the advisory and normalized package, explain why no compatible fix is available,
link a RingRift tracking issue, record an approval date, and expire no more than 45 days later.

The wrapper fails when an exception is malformed, duplicated, future-dated, expired, longer than
45 days, no longer used by the current audit, or attached to an advisory that reports a fix. It
also fails on unknown findings and pip-audit tool or dependency-resolution errors. This makes an
exception a short review window, not a permanent audit bypass.

## Review procedure

1. Upgrade to a compatible fixed release whenever one exists.
2. If no fixed release exists, document the exposure and mitigation in the rationale and link an
   issue that tracks replacement, isolation, or upstream release monitoring.
3. Keep the approval-to-expiry interval at 45 days or less.
4. Before expiry, remove the exception after upgrading or renew it through a reviewed change with
   current evidence and a new approval window.

As of 2026-07-17, the sole exception is `PYSEC-2026-1325` in the transitive `ecdsa` package used by
`p2pd`. The audit reported no fixed release; RingRift issue
[#113](https://github.com/synaptent/RingRift/issues/113) tracks replacement or isolation before
the exception expires.
