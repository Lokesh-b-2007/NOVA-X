"""Tests for FastAPI startup initialization."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import main  # noqa: E402


def test_app_lifespan_initializes_database_once(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "init_db", lambda: calls.append("initialized"))

    async def run_lifespan():
        async with main.lifespan(main.app):
            assert calls == ["initialized"]
        assert calls == ["initialized"]

    asyncio.run(run_lifespan())
