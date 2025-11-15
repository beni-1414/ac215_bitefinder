from __future__ import annotations

from api import config


def test_settings_defaults():
    s = config.settings
    assert hasattr(s, "INPUT_EVAL_URL")
    assert hasattr(s, "VL_MODEL_URL")
    assert hasattr(s, "RAG_MODEL_URL")
    # Defaults are strings
    assert isinstance(s.INPUT_EVAL_URL, str)
    assert isinstance(s.VL_MODEL_URL, str)
    assert isinstance(s.RAG_MODEL_URL, str)
