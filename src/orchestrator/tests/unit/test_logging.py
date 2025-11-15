from __future__ import annotations

from api.logging import setup_logging, get_logger


def test_setup_logging_returns_logger():
    logger = setup_logging()
    assert logger is not None
    g = get_logger()
    assert hasattr(g, "info")
    # call a method to ensure it doesn't raise
    g.info("test-log", test_key="value")
