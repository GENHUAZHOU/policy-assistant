import json

import pytest

from app.config import Settings
from app.security import (
    ToolContext,
    ToolPermissionError,
    ToolValidationError,
    check_permission,
    mask_secrets,
    validate_path,
    validate_reimbursement,
)
from app.tools import invoke_tool


def test_permission_matrix():
    check_permission("admin", "send_notification")
    check_permission("query", "search_policy")
    with pytest.raises(ToolPermissionError):
        check_permission("query", "submit_reimbursement")
    with pytest.raises(ToolPermissionError):
        check_permission("write", "send_notification")


def test_path_traversal_blocked(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text("ok", encoding="utf-8")
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    assert validate_path("policy.md", [docs]).name == "policy.md"
    with pytest.raises(ToolValidationError):
        validate_path("../secret.md", [docs])
    with pytest.raises(ToolValidationError):
        validate_path(str(tmp_path / "secret.md"), [docs])


def test_reimbursement_validation():
    with pytest.raises(ToolValidationError):
        validate_reimbursement({"amount": 0, "date": "2026-01-01", "category": "差旅", "reason": "x"}, 2000)
    with pytest.raises(ToolValidationError):
        validate_reimbursement({"amount": 10, "date": "not-a-date", "category": "差旅", "reason": "x"}, 2000)
    cleaned, needs_approval, _ = validate_reimbursement(
        {"amount": 2500, "date": "2026-01-01", "category": "差旅", "reason": "出差住宿"}, 2000
    )
    assert cleaned["amount"] == 2500.0
    assert needs_approval is True
    _, needs_approval, _ = validate_reimbursement(
        {"amount": 100, "date": "2026-01-01", "category": "差旅", "reason": "出差住宿"}, 2000
    )
    assert needs_approval is False


def test_mask_secrets():
    masked = mask_secrets("key=sk-abcdefghijklmnop 身份证 110101199001011234")
    assert "sk-abcdefghijklmnop" not in masked
    assert "110101199001011234" not in masked
    assert "[REDACTED]" in masked


def test_denied_tool_is_audited(settings: Settings):
    with pytest.raises(ToolPermissionError):
        invoke_tool(
            "send_notification",
            {"channel": "email", "message": "test"},
            ToolContext(thread_id="t1", role="query"),
            settings=settings,
        )
    lines = [json.loads(line) for line in settings.audit_log.read_text(encoding="utf-8").splitlines()]
    assert any(line["event"] == "tool_denied" and line["tool"] == "send_notification" for line in lines)
