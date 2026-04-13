"""Unit tests for coordination modules.

Test Coverage Strategy (December 2025)
======================================

This directory contains 220+ test files covering 99.5% of coordination modules.

Modules WITHOUT Direct Tests (by design):
-----------------------------------------
1. `__init__.py` - Public API re-exports (tested via consumers)
2. `_exports_core.py` - Internal exports (tested via consumers)
3. `_exports_daemon.py` - Internal exports (tested via consumers)
4. `_exports_events.py` - Internal exports (tested via consumers)
5. `_exports_orchestrators.py` - Internal exports (tested via consumers)
6. `_exports_sync.py` - Internal exports (tested via consumers)
7. `_exports_utils.py` - Internal exports (tested via consumers)
8. `handler_base.py` - Canonical handler API (tested directly)

These export modules are pure re-exports with no logic - they're tested
transitively when their consumers import from them. Adding direct tests
would be redundant and increase maintenance burden without value.

Test Organization:
-----------------
- Each coordination module has a corresponding `test_<module>.py` file
- Integration tests are in `tests/integration/coordination/`
- Tests use pytest fixtures from `conftest.py`
- Async tests use `@pytest.mark.asyncio` decorator

Running Tests:
--------------
    pytest tests/unit/coordination/ -v          # All coordination tests
    pytest tests/unit/coordination/test_*.py    # Specific patterns
    pytest -k "daemon" tests/unit/coordination/ # Keyword filtering
"""
