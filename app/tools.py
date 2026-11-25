"""工具实现与唯一执行入口 invoke_tool（权限 -> 校验 -> 审批 -> 执行 -> 审计）。"""
from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Callable

from app.approvals import ApprovalStore
from app.config import Settings
from app.rag.retriever import HybridRetriever
from app.security import (
    ApprovalRequired,
    ToolContext,
    ToolPermissionError,
    ToolValidationError,
    audit,
    check_permission,
    validate_notification,
    validate_path,
    validate_reimbursement,
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_policy",
        "description": "检索企业制度条款，返回带出处与生效日期的原文片段。回答制度问题必须调用。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要查询的制度问题"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_doc",
        "description": "读取允许目录内的制度文档原文，路径禁止 .. 跳转。",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对 docs 目录的文件路径"}},
            "required": ["path"],
        },
    },
    {
        "name": "submit_reimbursement",
        "description": "提交报销申请；金额超过阈值时必须人工审批。",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "报销金额（元）"},
                "category": {"type": "string", "description": "费用类别"},
                "date": {"type": "string", "description": "费用发生日期 YYYY-MM-DD"},
                "reason": {"type": "string", "description": "报销事由"},
            },
            "required": ["amount", "category", "date", "reason"],
        },
    },
    {
        "name": "send_notification",
        "description": "向员工发送通知，仅管理员可用。",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": ["email", "wecom", "feishu"]},
                "message": {"type": "string", "description": "通知内容"},
            },
            "required": ["channel", "message"],
        },
    },
]


def _log_business(settings: Settings, kind: str, record: dict[str, Any]) -> str:
    settings.business_log.parent.mkdir(parents=True, exist_ok=True)
    with settings.business_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": kind, **record}, ensure_ascii=False) + "\n")
    return record["id"]


def _execute(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    settings: Settings,
    retriever: HybridRetriever | None,
) -> dict[str, Any]:
    if name == "search_policy":
        if retriever is None:
            raise ToolValidationError("检索服务未初始化")
        results = retriever.search(str(args["query"]), top_k=settings.retrieval_top_k)
        context, citations = retriever.context(results)
        return {
            "documents": results,
            "context": context,
            "citations": citations,
            "count": len(results),
        }
    if name == "read_doc":
        path = validate_path(str(args["path"]), [Path(settings.docs_dir)])
        return {"path": str(path.relative_to(Path(settings.docs_dir).resolve())), "content": path.read_text(encoding="utf-8", errors="replace")[:8000]}
    if name == "submit_reimbursement":
        record = {
            "id": f"RB-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
            "thread_id": ctx.thread_id,
            "applicant": ctx.user_id,
            **args,
        }
        _log_business(settings, "reimbursement", record)
        return {"status": "submitted", "id": record["id"], "amount": record["amount"]}
    if name == "send_notification":
        record = {"id": f"NT-{uuid.uuid4().hex[:8].upper()}", **args, "sender": ctx.user_id}
        _log_business(settings, "notification", record)
        return {"status": "notified", "id": record["id"], "channel": record["channel"]}
    raise ToolValidationError(f"未知工具：{name}")


def _validate(name: str, args: dict[str, Any], settings: Settings) -> tuple[dict[str, Any], bool, str]:
    if name == "search_policy":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolValidationError("查询内容不能为空")
        return {"query": query[:1000]}, False, ""
    if name == "read_doc":
        path = str(args.get("path", "")).strip()
        if not path:
            raise ToolValidationError("文件路径不能为空")
        return {"path": path}, False, ""
    if name == "submit_reimbursement":
        return validate_reimbursement(args, settings.approval_threshold)
    if name == "send_notification":
        return validate_notification(args), False, ""
    raise ToolValidationError(f"未知工具：{name}")


def _summary(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if name == "search_policy":
        return {"count": result.get("count"), "citations": result.get("citations")}
    if name == "read_doc":
        return {"path": result.get("path"), "length": len(result.get("content", ""))}
    return {"id": result.get("id"), "status": result.get("status")}


def invoke_tool(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    settings: Settings,
    retriever: HybridRetriever | None = None,
    approvals: ApprovalStore | None = None,
    tool_call_id: str = "",
    decision: dict[str, Any] | None = None,
    resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """唯一工具执行入口；越权直接拦截并审计，敏感工具先走审批。"""
    try:
        check_permission(ctx.role, name)
    except ToolPermissionError as exc:
        audit(settings, "tool_denied", thread_id=ctx.thread_id, user_id=ctx.user_id, role=ctx.role, tool=name, error=str(exc))
        raise

    cleaned, requires_approval, reason = _validate(name, args, settings)

    if requires_approval:
        if approvals is None:
            raise ToolValidationError("审批存储未初始化")
        approval_id = f"{ctx.thread_id}:{tool_call_id or 'manual'}"
        record = approvals.get(approval_id) or approvals.create(
            approval_id, ctx.thread_id, name, cleaned, reason or "敏感操作需人工确认"
        )
        if record["status"] == "pending":
            if decision is None and resolver is None:
                audit(settings, "approval_required", thread_id=ctx.thread_id, tool=name, approval_id=approval_id, reason=reason)
                raise ApprovalRequired(record)
            if decision is None and resolver is not None:
                audit(settings, "approval_required", thread_id=ctx.thread_id, tool=name, approval_id=approval_id, reason=reason)
                decision = resolver(record)
            if decision is not None:
                record = approvals.decide(
                    approval_id, bool(decision.get("approved")), str(decision.get("approver") or "admin")
                )
        if not record or record["status"] != "approved":
            audit(settings, "approval_rejected", thread_id=ctx.thread_id, tool=name, approval_id=approval_id)
            return {"status": "rejected", "approval_id": approval_id, "message": "审批未通过，操作已取消"}

    result = _execute(name, cleaned, ctx, settings, retriever)
    audit(
        settings,
        "tool_executed",
        thread_id=ctx.thread_id,
        user_id=ctx.user_id,
        role=ctx.role,
        tool=name,
        args=cleaned,
        approved=requires_approval,
        result=_summary(name, result),
    )
    return result
