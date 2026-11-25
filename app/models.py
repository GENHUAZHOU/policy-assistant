"""模型适配层：有 API Key 时走 OpenAI 兼容接口，否则使用离线规则模型。

离线模型只用于本地开发与评测，它仍然遵循真实的 tool_call 协议，
因此上层图、权限、审批和审计逻辑与生产模式完全一致。
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings

_REMOTE_EMBEDDING_FAILED = False
_REMOTE_LLM_FAILED = False

POLICY_KEYWORDS = ("制度", "规定", "政策", "标准", "流程", "条款", "多少", "如何", "怎么", "是否", "policy", "rule")
NOTIFY_KEYWORDS = ("通知", "发消息", "发送通知", "notify", "notification")
REIMBURSE_KEYWORDS = ("报销", "报销申请", "reimbursement", "费用申请")


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


class LocalHashEmbeddings(Embeddings):
    """字符 bigram 哈希向量：确定性、无网络、适合小语料演示。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        for seg in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend(seg)
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        return tokens

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class LocalRuleChatModel(BaseChatModel):
    """离线规则模型：按关键词发起工具调用，并用工具结果生成可溯源答复。"""

    tools: list[Any] = []
    model_name: str = "local-rule"

    @property
    def _llm_type(self) -> str:
        return "local-rule"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "LocalRuleChatModel":
        return self.model_copy(update={"tools": list(tools)})

    @staticmethod
    def _extract_amount(text: str) -> float | None:
        match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*元", text)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None

    def _tool_call(self, name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            id=f"ai_{uuid.uuid4().hex[:8]}",
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}],
        )

    def _final_from_tools(self, messages: list[BaseMessage]) -> AIMessage:
        documents: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            status = payload.get("status")
            if status == "rejected":
                return AIMessage("该操作已被拒绝，未执行任何变更。")
            if status == "submitted":
                return AIMessage(
                    f"报销申请已提交，单号 {payload.get('id')}，金额 {payload.get('amount')} 元。"
                    "可在审批页查看状态。"
                )
            if status == "notified":
                return AIMessage(f"通知已通过 {payload.get('channel')} 发送。")
            if isinstance(payload.get("documents"), list):
                documents = payload["documents"]

        if not documents:
            return AIMessage("制度未提及该事项，建议咨询制度归口部门。")

        doc = documents[0]
        citation = doc.get("source") or f"{doc.get('doc_id')}#{doc.get('clause', '-')}"
        text = str(doc.get("text", "")).strip()
        return AIMessage(f"{text[:260]}\n\n（依据：{citation}）")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        human_texts = [_text_of(m) for m in messages if isinstance(m, HumanMessage)]
        latest = human_texts[-1] if human_texts else ""
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)

        if has_tool_result:
            response = self._final_from_tools(messages)
        elif any(word in latest for word in NOTIFY_KEYWORDS):
            response = self._tool_call("send_notification", {"channel": "email", "message": latest})
        elif any(word in latest for word in REIMBURSE_KEYWORDS):
            amount = self._extract_amount(latest)
            response = self._tool_call(
                "submit_reimbursement",
                {
                    "amount": amount if amount is not None else 0,
                    "category": "其他",
                    "date": date.today().isoformat(),
                    "reason": latest[:120],
                },
            )
        elif any(word in latest.lower() for word in POLICY_KEYWORDS):
            response = self._tool_call("search_policy", {"query": latest})
        else:
            response = AIMessage("我可以查询制度条款或提交报销申请，请描述你的问题。")

        return ChatResult(generations=[ChatGeneration(message=response)])


def build_embeddings(settings: Settings) -> Embeddings:
    global _REMOTE_EMBEDDING_FAILED
    if settings.openai_api_key and not _REMOTE_EMBEDDING_FAILED:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        try:
            embeddings.embed_query("connectivity probe")
        except Exception as exc:
            _REMOTE_EMBEDDING_FAILED = True
            print(f"[warn] 远端 Embedding 不可用，降级到本地哈希向量：{exc}")
        else:
            return embeddings
    if _REMOTE_EMBEDDING_FAILED:
        settings.embedding_dim_override = 256
    return LocalHashEmbeddings(dim=settings.embedding_dim)


def build_chat_model(settings: Settings) -> BaseChatModel:
    global _REMOTE_LLM_FAILED
    if settings.openai_api_key and not _REMOTE_LLM_FAILED:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        try:
            model.invoke("ping")
        except Exception as exc:
            _REMOTE_LLM_FAILED = True
            print(f"[warn] 远端 LLM 不可用，降级到本地规则模型：{exc}")
        else:
            return model
    return LocalRuleChatModel()
