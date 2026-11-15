"""Fail-closed Vision API OCR request counter shared across all OCR workers.

Budget: at most 800 reservations in ANY rolling 32 days (stricter than monthly).
A reservation is counted even if the Google call subsequently fails or times out.
Requires Redis persistence across restarts and exclusivity of the Vision project.
"""
from __future__ import annotations

import os
import time
import uuid

OCR_KEY = "smarthealthdog:vision:document_text:reservations:v1"
LIMIT = 800
WINDOW_SECONDS = 32 * 24 * 60 * 60

# Atomic across concurrent workers; prune obsolete entries and reserve a slot.
_RESERVE_SCRIPT = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
local used = redis.call('ZCARD', KEYS[1])
if used >= tonumber(ARGV[3]) then return 0 end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return 1
"""

class OCRQuotaExceeded(Exception):
    pass

class OCRQuotaUnavailable(Exception):
    pass


def reserve_document_text_unit() -> None:
    """One request = one IMAGE DOCUMENT_TEXT_DETECTION unit (not PDF)."""
    try:
        from redis import Redis
        configured = int(os.getenv("OCR_MAX_REQUESTS_32D", str(LIMIT)))
        if not 1 <= configured <= LIMIT:
            raise ValueError("OCR_MAX_REQUESTS_32D must be between 1 and 800")
        client = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - WINDOW_SECONDS * 1000
        allowed = client.eval(
            _RESERVE_SCRIPT, 1, OCR_KEY,
            str(cutoff_ms), str(now_ms), str(configured),
            uuid.uuid4().hex, str(WINDOW_SECONDS + 86400),
        )
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        raise OCRQuotaUnavailable("OCR quota guard configuration/Redis error") from exc
    except Exception as exc:
        # If Redis, credentials or Lua fail, NEVER call the billable API.
        raise OCRQuotaUnavailable("OCR quota guard unavailable") from exc

    if allowed != 1:
        raise OCRQuotaExceeded("OCR free-tier safety cap reached")
