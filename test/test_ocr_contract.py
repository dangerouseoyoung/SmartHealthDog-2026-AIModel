"""Offline tests: no Google credentials, Redis server, or real HTTP calls required."""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocr.health_check_parser import HealthCheckParser
from ocr import quota_guard as ocr_quota_guard


def textblock(text, x1, y1, x2=None, y2=None, confidence=0.94):
    x2 = x2 if x2 is not None else x1 + max(10, len(text) * 13)
    y2 = y2 if y2 is not None else y1 + 20
    return {
        "text": text, "confidence": confidence,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": (x1 + x2) // 2, "center_y": (y1 + y2) // 2,
        "bbox": [],
    }


def make_blocks():
    items = [
        ("종류", "개"), ("품종", "말티즈"), ("동물명", "초코"),
        ("동물등록번호", "123456789012345"), ("성별", "수컷"),
        ("연령", "3세"), ("모색", "흰색"), ("특징", "흰털"),
        ("병명", "피부염"), ("발병일", "2026.10.20"),
        ("진단일", "2026.11.10"), ("주요 증상", "발적"),
        ("치료내용", "약 처방"), ("입퇴원일", "2026.11.10 ~ 2026.11.11"),
        ("예후", "양호"), ("비고", "재검"), ("작성일", "2026.11.12"),
    ]
    result = []
    for index, (label, val) in enumerate(items):
        y = index * 70
        result.extend((textblock(label, 20, y, 120), textblock(val, 220, y, 600)))
    return result


def test_parser_dates_and_no_invalid_dates():
    parser = HealthCheckParser()
    assert parser.normalize_date("2026년 11월 15일") == "2026-11-15"
    assert parser.normalize_date("2026년 2월 30일") is None
    assert parser.normalize_date("2024.02.29") == "2024-02-29"
    assert parser.normalize_date("2026-13-01") is None


def test_parser_structure_and_hospitalization_range():
    fields = HealthCheckParser().parse({"blocks": make_blocks()})
    assert len(fields) == 17
    assert fields["animal_name"] == "초코"
    assert fields["disease_name"] == "피부염"
    assert fields["diagnosis_date"] == "2026-11-10"
    assert fields["hospitalization_discharge_date"] == "2026-11-10 ~ 2026-11-11"


def load_task_module():
    # Stub external integrations to test the data-contract path in isolation.
    import importlib
    fake_celery = types.ModuleType("celery_app")
    fake_celery.app = types.SimpleNamespace(task=lambda **kwargs: (lambda func: func))
    fake_vision = types.ModuleType("ocr.google_vision_ocr")
    fake_vision.extract_document = lambda data: None
    with patch.dict(sys.modules, {"celery_app": fake_celery, "ocr.google_vision_ocr": fake_vision}):
        sys.modules.pop("ocr.certificate_ocr_task", None)
        return importlib.import_module("ocr.certificate_ocr_task")


def test_backend_contract_has_17_camelcase_fields_and_metadata():
    task = load_task_module()
    result = {"success": True, "rawText": "진단서\\n병명: 피부염", "blocks": make_blocks(), "error": None}
    with patch.object(task, "extract_document", return_value=result):
        payload = task.process_certificate(b"mock-image")
    assert payload["success"] is True
    assert payload["rawText"] == result["rawText"]
    assert isinstance(payload["confidence"], float) and payload["confidence"] == 0.94
    assert len(payload["fields"]) == 17
    assert payload["fields"]["animalName"] == "초코"
    assert payload["fields"]["diseaseName"] == "피부염"
    assert payload["fields"]["diagnosedDateText"] == "2026-11-10"
    assert payload["fields"]["hospitalizationDischargeDateText"] == "2026-11-10 ~ 2026-11-11"
    assert "animal_name" not in payload["fields"]


def test_failure_result_is_compatible():
    task = load_task_module()
    with patch.object(task, "extract_document", return_value={"success": False, "error": "OCR_MONTHLY_LIMIT"}):
        payload = task.process_certificate(b"test")
    assert payload == {"success": False, "fields": None, "error": "OCR_MONTHLY_LIMIT"}


class FakeRedis:
    entries = {}
    unavailable = False

    def __init__(self, **kwargs):
        pass

    def eval(self, script, numkeys, key, cutoff, now, limit, member, ttl):
        if self.unavailable:
            raise ConnectionError("simulated redis failure")
        cutoff, now, limit = int(cutoff), int(now), int(limit)
        entries = self.entries.setdefault(key, {})
        for k, v in list(entries.items()):
            if v <= cutoff:
                entries.pop(k)
        if len(entries) >= limit:
            return 0
        entries[member] = now
        return 1


def test_quota_guard_atomic_cap_and_fail_closed():
    FakeRedis.entries.clear()
    FakeRedis.unavailable = False
    fake_redis_module = types.SimpleNamespace(Redis=FakeRedis)
    with patch.dict(sys.modules, {"redis": fake_redis_module}), patch.dict("os.environ", {"OCR_MAX_REQUESTS_32D": "2"}):
        ocr_quota_guard.reserve_document_text_unit()
        ocr_quota_guard.reserve_document_text_unit()
        with pytest.raises(ocr_quota_guard.OCRQuotaExceeded):
            ocr_quota_guard.reserve_document_text_unit()
        FakeRedis.unavailable = True
        with pytest.raises(ocr_quota_guard.OCRQuotaUnavailable):
            ocr_quota_guard.reserve_document_text_unit()
    FakeRedis.unavailable = False


def test_quota_misconfiguration_fails_closed():
    fake_redis_module = types.SimpleNamespace(Redis=FakeRedis)
    with patch.dict(sys.modules, {"redis": fake_redis_module}), patch.dict("os.environ", {"OCR_MAX_REQUESTS_32D": "801"}):
        with pytest.raises(ocr_quota_guard.OCRQuotaUnavailable):
            ocr_quota_guard.reserve_document_text_unit()


def test_billable_vision_call_never_runs_when_cap_reached():
    import importlib
    google = types.ModuleType("google")
    google.__path__ = []
    api_core = types.ModuleType("google.api_core")
    api_core.__path__ = []
    errors = types.ModuleType("google.api_core.exceptions")
    for name in ("Unauthenticated", "PermissionDenied", "DeadlineExceeded", "ResourceExhausted", "TooManyRequests"):
        setattr(errors, name, type(name, (Exception,), {}))
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    vision = types.ModuleType("google.cloud.vision")
    vision.Image = lambda **kwargs: kwargs
    vision.ImageContext = lambda **kwargs: kwargs
    google.api_core = api_core
    google.cloud = cloud
    api_core.exceptions = errors
    cloud.vision = vision
    modules = {
        "google": google, "google.api_core": api_core,
        "google.api_core.exceptions": errors,
        "google.cloud": cloud, "google.cloud.vision": vision,
    }
    with patch.dict(sys.modules, modules):
        sys.modules.pop("ocr.google_vision_ocr", None)
        mod = importlib.import_module("ocr.google_vision_ocr")
        picture = Image.new("RGB", (80, 30), "white")
        buff = io.BytesIO()
        picture.save(buff, format="PNG")
        class FakeClient:
            called = False
            def document_text_detection(self, **kwargs):
                self.called = True
                raise AssertionError("Vision should never be called")
        client = FakeClient()
        with patch.object(mod, "reserve_document_text_unit", side_effect=mod.OCRQuotaExceeded):
            result = mod.extract_document(buff.getvalue(), client=client)
        assert result["error"] == "OCR_MONTHLY_LIMIT"
        assert not client.called
    sys.modules.pop("ocr.google_vision_ocr", None)  # stub google 모듈로 로드된 사본 제거
