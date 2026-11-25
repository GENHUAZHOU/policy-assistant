"""LangGraph 组装：节点 + 条件边 + checkpoint。"""
from contextlib import ExitStack
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.approvals import ApprovalStore
from app.config import Settings, get_settings
from app.models import build_chat_model
from app.nodes import AgentState, NodeDeps, agent_node, approval_node, retrieve_node, route_after_agent, tools_node
from app.observability import BudgetGate, Tracer
from app.rag.retriever import HybridRetriever, create_retriever

_PG_STACKS: list[ExitStack] = []


def _checkpointer(settings: Settings) -> Any:
    if settings.database_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            stack = ExitStack()
            saver = stack.enter_context(PostgresSaver.from_conn_string(settings.database_url))
            saver.setup()
            _PG_STACKS.append(stack)
            return saver
        except Exception as exc:  # pragma: no cover - 取决于外部数据库
            print(f"[warn] Postgres checkpoint 不可用，降级到 MemorySaver：{exc}")
    return MemorySaver()


def create_graph(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    retriever: HybridRetriever | None = None,
    approvals: ApprovalStore | None = None,
    tracer: Tracer | None = None,
) -> Any:
    settings = settings or get_settings()
    deps = NodeDeps(
        settings=settings,
        model=model or build_chat_model(settings),
        retriever=retriever or create_retriever(settings),
        approvals=approvals or ApprovalStore(settings.approval_log),
        tracer=tracer or Tracer(settings),
        gate=BudgetGate(settings.token_budget, settings.max_turns),
    )

    def agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        return agent_node(state, config, deps=deps)

    def retrieve(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        return retrieve_node(state, config, deps=deps)

    def tools(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        return tools_node(state, config, deps=deps)

    def approval(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        return approval_node(state, config, deps=deps)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("tools", tools)
    builder.add_node("approval", approval)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"retrieve": "retrieve", "tools": "tools", "approval": "approval", END: END},
    )
    for node in ("retrieve", "tools", "approval"):
        builder.add_edge(node, "agent")
    return builder.compile(checkpointer=_checkpointer(settings))
