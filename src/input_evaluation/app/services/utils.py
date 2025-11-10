from __future__ import annotations
import time
import re
from typing import Iterable
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        dur = int((time.perf_counter() - start) * 1000)
        return result, dur
    return wrapper