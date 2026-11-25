import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.approvals import ApprovalStore
from app.config import Settings
from app.graph import create_graph
from app.models import LocalHashEmbeddings, LocalRuleChatModel
from app.rag.ingest import ingest
from app.rag.retriever import HybridRetriever
from app.rag.store import LocalVectorStore


@pytest.fixture
def env(settings: Settings):
    embeddings = LocalHashEmbeddings(settings.embedding_dim)
    store = LocalVectorStore(settings.index_dir)
    ingest(settings, embeddings, store)
    retriever = HybridRetriever(settings, embeddings, store)
    approvals = ApprovalStore(settings.approval_log)
    graph = create_graph(settings, model=LocalRuleChatModel(), retriever=retriever, approvals=approvals)
    return graph, approvals


def config(thread_id: str, role: str = "write") -> dict:
    return {"configurable": {"thread_id": thread_id, "role": role, "user_id": "tester"}}


def payload(message: str) -> dict:
    return {"messages": [HumanMessage(content=message)], "citations": [], "pending_approval": None, "tokens_used": 0}


def last_answer(result: dict) -> str:
    return next(message.content for message in reversed(result["messages"]) if isinstance(message, AIMessage))


def test_policy_question_uses_rag(env):
    graph, _ = env
    result = graph.invoke(payload("一线城市住宿费上限是多少？"), config("t-rag", "query"))
    answer = last_answer(result)
    assert "500" in answer
    assert "依据" in answer
    assert any(item["source"].startswith("报销管理制度#") for item in result["citations"])


def test_approval_then_execute(env):
    graph, approvals = env
    cfg = config("t-approve")
    graph.invoke(payload("我要报销住宿费2500元，类别差旅，事由出差住宿"), cfg)
    assert graph.get_state(cfg).interrupts
    result = graph.invoke(Command(resume={"approved": True, "approver": "admin"}), cfg)
    assert "已提交" in last_answer(result)
    assert any(record["status"] == "approved" for record in approvals.list_all())


def test_approval_rejected(env):
    graph, approvals = env
    cfg = config("t-reject")
    graph.invoke(payload("我要报销住宿费3000元，类别差旅，事由出差住宿"), cfg)
    result = graph.invoke(Command(resume={"approved": False, "approver": "admin"}), cfg)
    assert "拒绝" in last_answer(result)
    assert any(record["status"] == "rejected" for record in approvals.list_all())


def test_low_amount_skips_approval(env):
    graph, approvals = env
    cfg = config("t-small")
    result = graph.invoke(payload("我要报销市内交通100元，类别交通，事由客户拜访"), cfg)
    assert "已提交" in last_answer(result)
    assert not graph.get_state(cfg).interrupts
    assert approvals.list_all() == []


def test_budget_gate(settings: Settings):
    tight = settings.with_overrides(token_budget=1)
    embeddings = LocalHashEmbeddings(tight.embedding_dim)
    store = LocalVectorStore(tight.index_dir)
    ingest(tight, embeddings, store)
    graph = create_graph(
        tight,
        model=LocalRuleChatModel(),
        retriever=HybridRetriever(tight, embeddings, store),
        approvals=ApprovalStore(tight.approval_log),
    )
    result = graph.invoke(payload("差旅住宿标准是多少？"), config("t-budget", "query"))
    assert "预算" in last_answer(result)
