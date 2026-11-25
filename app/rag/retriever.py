"""混合检索：向量 + BM25 -> RRF 合并 -> rerank -> 每轮裁剪。"""
from __future__ import annotations

import html
import math
import re
from collections import Counter
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.models import build_embeddings
from app.rag.store import Chunk, VectorStore, create_vector_store

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
GENERIC_BIGRAMS = {
    "公司", "员工", "费用", "报销", "制度", "规定", "标准", "多少", "怎么", "如何", "是否",
    "可以", "需要", "什么", "我们", "一下", "请问", "申请", "多久", "相关", "哪些", "一个",
    "提供", "使用", "关于", "包括", "以下", "进行",
}
SYNONYM_EXPANSION = {
    "酒店": "住宿", "住店": "住宿", "旅馆": "住宿", "过期": "逾期", "期限": "时限",
}


def tokenize(text: str) -> list[str]:
    """中文按字+bigram、英文按词切分，供 BM25 使用。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(seg)
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens


def content_bigrams(query: str) -> set[str]:
    """去掉“公司/报销/多少”等通用词后，问题中的实质性中文bigram。"""
    terms = {
        seg[i : i + 2]
        for seg in re.findall(r"[\u4e00-\u9fff]+", query.lower())
        for i in range(len(seg) - 1)
    }
    terms -= GENERIC_BIGRAMS
    for colloquial, canonical in SYNONYM_EXPANSION.items():
        if colloquial in query:
            terms.add(canonical)
    return terms


class HybridRetriever:
    def __init__(self, settings: Settings, embeddings: Embeddings, store: VectorStore) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self._reranker = None
        if settings.reranker_model:
            try:
                from FlagEmbedding import FlagReranker

                self._reranker = FlagReranker(settings.reranker_model, use_fp16=False)
            except Exception as exc:  # pragma: no cover - 取决于外部模型
                print(f"[warn] reranker 加载失败，使用词法重排：{exc}")

    def _bm25_scores(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        doc_tokens = [tokenize(chunk["text"]) for chunk in chunks]
        lengths = [len(tokens) for tokens in doc_tokens]
        avgdl = sum(lengths) / len(lengths) or 1.0
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        total = len(chunks)
        idf = {
            term: 1.0 + math.log((total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        scores = []
        for tokens in doc_tokens:
            freq = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in set(tokenize(query)):
                term_freq = freq.get(term, 0)
                if not term_freq:
                    continue
                denominator = term_freq + BM25_K1 * (1 - BM25_B + BM25_B * length / avgdl)
                score += idf[term] * term_freq * (BM25_K1 + 1) / denominator
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """返回带 score/source 的条款片段；索引为空或检索异常时返回空列表。"""
        top_k = top_k or self.settings.retrieval_top_k
        try:
            chunks = self.store.all()
            if not chunks or not query.strip():
                return []

            query_vector = self.embeddings.embed_query(query)
            vector_hits = self.store.search(query_vector, top_k=max(top_k * 3, 10))
            bm25_scores = self._bm25_scores(query, chunks)

            rrf: dict[str, float] = {}
            for rank, (chunk, _) in enumerate(vector_hits, start=1):
                rrf[chunk["id"]] = rrf.get(chunk["id"], 0.0) + 1.0 / (RRF_K + rank)
            bm25_order = sorted(range(len(chunks)), key=lambda i: -bm25_scores[i])
            for rank, index in enumerate(bm25_order[: max(top_k * 3, 10)], start=1):
                if bm25_scores[index] <= 0:
                    continue
                chunk_id = chunks[index]["id"]
                rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if not rrf:
                return []

            by_id = {chunk["id"]: chunk for chunk in chunks}
            candidates = [
                (by_id[chunk_id], score)
                for chunk_id, score in rrf.items()
                if chunk_id in by_id
            ]

            # 幻觉抑制：问题必须包含至少一个与条款实质相关的中文词，纯通用词不作答。
            terms = content_bigrams(query)
            if terms:
                candidates = [
                    (chunk, score)
                    for chunk, score in candidates
                    if any(term in chunk["text"] for term in terms)
                ]
                if not candidates:
                    return []

            if self._reranker is not None:  # pragma: no cover - 可选外部模型
                pairs = [[query, chunk["text"]] for chunk, _ in candidates]
                rerank_scores = self._reranker.compute_score(pairs, normalize=True)
                if not isinstance(rerank_scores, list):
                    rerank_scores = [rerank_scores]
                ranked = sorted(
                    zip(candidates, rerank_scores), key=lambda item: -float(item[1])
                )
                candidates = [pair for pair, _ in ranked]
            else:
                query_tokens = set(tokenize(query))
                lexical = {
                    chunk["id"]: len(query_tokens & set(tokenize(chunk["text"])))
                    for chunk, _ in candidates
                }
                candidates.sort(
                    key=lambda item: item[1] + 0.001 * lexical.get(item[0]["id"], 0),
                    reverse=True,
                )

            results = []
            for chunk, score in candidates[:top_k]:
                metadata = chunk.get("metadata", {})
                results.append(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": metadata,
                        "score": round(float(score), 6),
                        "source": f"{metadata.get('doc_id', 'unknown')}#{metadata.get('clause', '-')}",
                    }
                )
            return results
        except Exception as exc:  # 检索服务不可用时降级为空，由 agent 友好提示
            print(f"[warn] 检索失败：{exc}")
            return []

    def context(self, results: list[dict[str, Any]], prompt_k: int | None = None) -> tuple[str, list[dict[str, str]]]:
        """把结果包装成 untrusted_doc，并生成引用列表。"""
        prompt_k = prompt_k or self.settings.prompt_top_k
        selected = results[:prompt_k]
        if not selected:
            return "知识库中未找到相关规定。", []

        blocks = []
        citations: list[dict[str, str]] = []
        for item in selected:
            metadata = item["metadata"]
            source = html.escape(str(item["source"]), quote=True)
            attrs = " ".join(
                f'{key}="{html.escape(str(metadata.get(key, "")), quote=True)}"'
                for key in ("doc_id", "clause", "version", "effective_date")
            )
            blocks.append(f"<untrusted_doc source=\"{source}\" {attrs}>\n{item['text']}\n</untrusted_doc>")
            citations.append(
                {
                    "source": item["source"],
                    "doc_id": str(metadata.get("doc_id", "")),
                    "clause": str(metadata.get("clause", "")),
                    "version": str(metadata.get("version", "")),
                    "effective_date": str(metadata.get("effective_date", "")),
                }
            )
        return "\n\n".join(blocks), citations


def create_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(settings, build_embeddings(settings), create_vector_store(settings))
