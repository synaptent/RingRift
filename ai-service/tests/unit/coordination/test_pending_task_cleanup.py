from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_cleanup_fixture_tolerates_leaked_task():
    async def leaked_task():
        await asyncio.Event().wait()

    asyncio.create_task(leaked_task())
    await asyncio.sleep(0)
