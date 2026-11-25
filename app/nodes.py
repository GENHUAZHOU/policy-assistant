"""LangGraph 节点：agent / retrieve / tools / approval。"""
import json
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from app.approvals import ApprovalStore
from app.config import Settings
from app.observability import SYSTEM_PROMPT, BudgetGate, Tracer, estimate_tokens
from app.rag.retriever import HybridRetriever
from app.security import ApprovalRequired, ToolContext, mask_secrets
from app.tools import TOOL_SCHEMAS, invoke_tool

APPROVAL_TOOLS = {"submit_reimbursement"}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    citations: list[dict]
    pending_approval: dict | None
    tokens_used: int


@dataclass
class NodeDeps:
    settings: Settings
    model: BaseChatModel
    retriever: HybridRetriever
    approvals: ApprovalStore
    tracer: Tracer
    gate: BudgetGate


def _thread_id(config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable", {})
    return str(configurable.get("thread_id", "default"))


def _ctx(config: RunnableConfig | None) -> ToolContext:
    configurable = (config or {}).get("configurable", {})
    return ToolContext(
        thread_id=_thread_id(config),
        role=str(configurable.get("role", "query")),
        user_id=str(configurable.get("user_id", "anonymous")),
    )


def _latest_calls(state: AgentState) -> list[dict[str, Any]]:
    messages = state.get("messages", [])
    if not messages:
        return []
    last = messages[-1]
    if isinstance(last, AIMessage):
        return list(last.tool_calls or [])
    return []


def _count_turns(state: AgentState) -> int:
    return sum(1 for message in state.get("messages", []) if isinstance(message, AIMessage) and message.tool_calls)


def agent_node(state: AgentState, config: RunnableConfig | None = None, *, deps: NodeDeps) -> dict[str, Any]:
    thread_id = _thread_id(config)
    turns = _count_turns(state)
    reason = deps.gate.check(int(state.get("tokens_used", 0)), turns)
    if reason:
        return {"messages": [AIMessage(content=reason)], "citations": [], "pending_approval": None}

    messages = list(state.get("messages", []))
    if not any(isinstance(message, SystemMessage) for message in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    input_summary = messages[-1].content if messages else ""
    with deps.tracer.span("agent", thread_id, input_summary=str(input_summary)[:300]) as record:
        response = deps.model.bind_tools(TOOL_SCHEMAS).invoke(messages)
        record["output"] = f"tool_calls={[call['name'] for call in (response.tool_calls or [])]}"
    tokens = estimate_tokens(str(input_summary)) + estimate_tokens(str(response.content))
    return {"messages": [response], "tokens_used": int(state.get("tokens_used", 0)) + tokens}


def _tool_node(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    deps: NodeDeps,
    allow_approval: bool,
) -> dict[str, Any]:
    ctx = _ctx(config)
    thread_id = ctx.thread_id
    tool_messages: list[ToolMessage] = []
    citations = list(state.get("citations") or [])
    pending: RunnableConfig | None = None

    for call in _latest_calls(state):
        name = str(call.get("name", ""))
        args = dict(call.get("args") or {})
        tool_call_id = str(call.get("id", ""))
        resolver = None
        if allow_approval and name in APPROVAL_TOOLS:
            def resolver(payload: dict[str, Any]) -> dict[str, Any]:
                return interrupt({"kind": "approval", **payload})

        try:
            with deps.tracer.span("tool:" + name, thread_id, input_summary=args) as record:
                result = invoke_tool(
                    name,
                    args,
                    ctx,
                    settings=deps.settings,
                    retriever=deps.retriever,
                    approvals=deps.approvals,
                    tool_call_id=tool_call_id,
                    resolver=resolver,
                )
                record["output"] = {"status": result.get("status", "ok")}
        except (GraphInterrupt, GraphBubbleUp):
            raise
        except ApprovalRequired as exc:
            pending = exc.approval
            result = {"status": "approval_required", "message": "该操作需要人工审批，请等待管理员决定。"}
        except Exception as exc:
            result = {"status": "error", "message": mask_secrets(str(exc))}

        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
                name=name,
            )
        )
        for citation in result.get("citations", []) or []:
            if citation not in citations:
                citations.append(citation)

    return {"messages": tool_messages, "citations": citations, "pending_approval": pending}


def retrieve_node(state: AgentState, config: RunnableConfig | None = None, *, deps: NodeDeps) -> dict[str, Any]:
    return _tool_node(state, config, deps=deps, allow_approval=False)


def tools_node(state: AgentState, config: RunnableConfig | None = None, *, deps: NodeDeps) -> dict[str, Any]:
    return _tool_node(state, config, deps=deps, allow_approval=False)


def approval_node(state: AgentState, config: RunnableConfig | None = None, *, deps: NodeDeps) -> dict[str, Any]:
    return _tool_node(state, config, deps=deps, allow_approval=True)


def route_after_agent(state: AgentState) -> str:
    calls = _latest_calls(state)
    if not calls:
        return END
    names = {str(call.get("name", "")) for call in calls}
    if names & (APPROVAL_TOOLS | {"send_notification"}):
        return "approval"
    if "search_policy" in names:
        return "retrieve"
    return "tools"
