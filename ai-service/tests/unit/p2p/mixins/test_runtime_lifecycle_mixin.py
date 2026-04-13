from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from scripts.p2p.mixins.runtime_lifecycle_mixin import RuntimeLifecycleMixin


class _FakeSite:
    def __init__(self, effect=None) -> None:
        self._effect = effect
        self.stop = AsyncMock()

    async def start(self) -> None:
        if self._effect:
            raise self._effect


class _DummyRuntime(RuntimeLifecycleMixin):
    pass


@pytest.mark.asyncio
async def test_start_tcp_site_with_retry_recreates_site_after_addr_in_use() -> None:
    dummy = object.__new__(_DummyRuntime)
    runner = object()
    first = _FakeSite(OSError(98, "Address already in use"))
    second = _FakeSite()

    with patch(
        "scripts.p2p.mixins.runtime_lifecycle_mixin.web.TCPSite",
        side_effect=[first, second],
    ) as mock_site, patch(
        "scripts.p2p.mixins.runtime_lifecycle_mixin.asyncio.sleep",
        new=AsyncMock(),
    ) as mock_sleep:
        site = await dummy._start_tcp_site_with_retry(runner, "0.0.0.0", 8770)

    assert site is second
    assert mock_site.call_count == 2
    first.stop.assert_awaited_once()
    mock_sleep.assert_awaited_once()
