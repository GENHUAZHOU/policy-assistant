"""安全横切层：权限矩阵、参数校验、脱敏与审计。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings

ROLE_TOOLS: dict[str, set[str]] = {
    "query": {"search_policy", "read_doc"},
    "write": {"search_policy", "read_doc", "submit_reimbursement"},
    "admin": {"search_policy", "read_doc", "submit_reimbursement", "send_notification"},
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\b\d{17}[\dXx]\b"),
    re.compile(r"\b(?:\+?86)?1[3-9]\d{9}\b"),
    re.compile(r"\b\d{16,19}\b"),
]


class ToolPermissionError(PermissionError):
    pass


class ToolValidationError(ValueError):
    pass


class ApprovalRequired(Exception):
    """工具需要人工审批；payload 直接用于 graph.interrupt。"""

    def __init__(self, approval: dict[str, Any]) -> None:
        super().__init__(f"工具 {approval['tool']} 需要人工审批")
        self.approval = approval


@dataclass(frozen=True)
class ToolContext:
    thread_id: str
    role: str = "query"
    user_id: str = "anonymous"


def check_permission(role: str, tool_name: str) -> None:
    allowed = ROLE_TOOLS.get(role, set())
    if tool_name not in allowed:
        raise ToolPermissionError(f"角色 {role} 无权调用工具 {tool_name}")



def validate_path(raw_path: str, allowed_dirs: list[Path]) -> Path:
    path = Path(raw_path)
    if ".." in path.parts or path.is_absolute():
        raise ToolValidationError("非法路径：禁止绝对路径或 .. 跳转")
    candidates = [(Path(directory) / path).resolve() for directory in allowed_dirs]
    for candidate in candidates:
        if any(candidate.is_relative_to(Path(directory).resolve()) for directory in allowed_dirs):
            if candidate.is_file():
                return candidate
    raise ToolValidationError("文件不存在或不在允许目录内")


def validate_reimbursement(args: dict[str, Any], threshold: float) -> tuple[dict[str, Any], bool, str]:
    cleaned = dict(args)
    try:
        amount = float(cleaned.get("amount"))
    except (TypeError, ValueError) as exc:
        raise ToolValidationError("金额必须是数字") from exc
    if amount <= 0 or amount > 1_000_000:
        raise ToolValidationError("金额必须大于 0 且不超过 1000000 元")

    raw_date = str(cleaned.get("date", ""))
    try:
        parsed_date = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise ToolValidationError("日期必须为 YYYY-MM-DD") from exc
    if parsed_date > date.today():
        raise ToolValidationError("报销日期不能晚于今天")

    category = str(cleaned.get("category", "")).strip()
    reason = str(cleaned.get("reason", "")).strip()
    if not category or not reason:
        raise ToolValidationError("费用类别与事由不能为空")

    cleaned.update({"amount": round(amount, 2), "date": parsed_date.isoformat(), "category": category, "reason": reason[:500]})
    requires_approval = amount > threshold
    note = f"金额 {amount:.2f} 元超过审批阈值 {threshold:.2f} 元" if requires_approval else ""
    return cleaned, requires_approval, note


def validate_notification(args: dict[str, Any]) -> dict[str, Any]:
    channel = str(args.get("channel", "")).strip()
    message = str(args.get("message", "")).strip()
    if channel not in {"email", "wecom", "feishu"}:
        raise ToolValidationError("通知渠道必须是 email / wecom / feishu")
    if not message:
        raise ToolValidationError("通知内容不能为空")
    return {"channel": channel, "message": message[:1000]}


def mask_secrets(value: Any) -> Any:
    if isinstance(value, str):
        masked = value
        for pattern in SECRET_PATTERNS:
            masked = pattern.sub("[REDACTED]", masked)
        return masked
    if isinstance(value, dict):
        return {key: mask_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return value


def audit(settings: Settings, event: str, **fields: Any) -> None:
    """全量审计日志：谁、何时、调了什么、批没批。"""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **mask_secrets(fields),
    }
    settings.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with settings.audit_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
