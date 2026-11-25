"""FastAPI + SSE 接口：对话、审批续跑、审批页与运维接口。"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.approvals import ApprovalStore
from app.config import ROOT, get_settings
from app.graph import create_graph
from app.models import LocalRuleChatModel, build_chat_model, build_embeddings
from app.rag.ingest import ingest
from app.rag.retriever import create_retriever
from app.security import mask_secrets

app = FastAPI(title="企业制度知识助手", version="1.0.0")
_graph = None
_graph_lock = threading.Lock()


class ChatRequest(BaseModel):
    thread_id: str = "default"
    message: str
    role: str = "query"
    user_id: str = "anonymous"


class ResumeRequest(BaseModel):
    approved: bool
    approver: str = "admin"
    role: str = "query"


class DecideRequest(BaseModel):
    approved: bool
    approver: str = "admin"


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                settings = get_settings()
                retriever = create_retriever(settings)
                if retriever.store.count() == 0:
                    ingest(settings, build_embeddings(settings), retriever.store)
                _graph = create_graph(settings, retriever=retriever)
    return _graph


def _config(thread_id: str, role: str, user_id: str = "anonymous") -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id, "role": role, "user_id": user_id},
        "recursion_limit": 30,
    }


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _stream(payload: Any, config: dict[str, Any]) -> AsyncIterator[str]:
    graph = get_graph()
    emitted_interrupt = False
    emitted_text: set[str] = set()
    try:
        for chunk in graph.stream(payload, config=config, stream_mode="values"):
            interrupts = chunk.get("__interrupt__")
            if interrupts:
                emitted_interrupt = True
                yield _sse("interrupt", [getattr(item, "value", item) for item in interrupts])
            messages = chunk.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage) and last.content and last.id not in emitted_text:
                    emitted_text.add(last.id)
                    yield _sse("message", {"content": last.content, "citations": chunk.get("citations", [])})
        if not emitted_interrupt:
            state = graph.get_state(config)
            if state.interrupts:
                yield _sse("interrupt", [getattr(item, "value", item) for item in state.interrupts])
        yield _sse("done", {"ok": True})
    except Exception as exc:
        yield _sse("error", {"message": mask_secrets(str(exc))})


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    try:
        chunks = create_retriever(settings).store.count()
    except Exception:
        chunks = -1
    model = build_chat_model(settings)
    mode = "local" if isinstance(model, LocalRuleChatModel) else "openai"
    return {"status": "ok", "llm_mode": mode, "indexed_chunks": chunks}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    payload = {
        "messages": [HumanMessage(content=request.message)],
        "citations": [],
        "pending_approval": None,
        "tokens_used": 0,
    }
    return StreamingResponse(
        _stream(payload, _config(request.thread_id, request.role, request.user_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/threads/{thread_id}/resume")
async def resume(thread_id: str, request: ResumeRequest) -> StreamingResponse:
    decision = {"approved": request.approved, "approver": request.approver}
    return StreamingResponse(
        _stream(Command(resume=decision), _config(thread_id, request.role)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/approvals")
def list_approvals() -> list[dict[str, Any]]:
    settings = get_settings()
    store = ApprovalStore(settings.approval_log)
    return [mask_secrets(record) for record in store.list_all()]


@app.post("/api/approvals/{approval_id}/decide")
def decide_approval(approval_id: str, request: DecideRequest) -> dict[str, Any]:
    settings = get_settings()
    store = ApprovalStore(settings.approval_log)
    record = store.decide(approval_id, request.approved, request.approver)
    if record is None:
        raise HTTPException(status_code=404, detail="审批单不存在")
    return {"message": "审批已记录，请通过 /threads/{thread_id}/resume 续跑对话", "record": mask_secrets(record)}


@app.get("/admin/approvals")
def approval_page() -> FileResponse:
    return FileResponse(Path(ROOT) / "web" / "index.html")


@app.post("/admin/ingest")
def reingest() -> dict[str, Any]:
    settings = get_settings()
    retriever = create_retriever(settings)
    count = ingest(settings, build_embeddings(settings), retriever.store)
    global _graph
    with _graph_lock:
        _graph = None
    return {"chunks": count}


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=int(__import__("os").getenv("API_PORT", "8000")))


if __name__ == "__main__":
    main()
