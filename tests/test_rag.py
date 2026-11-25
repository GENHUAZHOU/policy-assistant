from app.config import Settings
from app.models import LocalHashEmbeddings
from app.rag.ingest import ingest
from app.rag.retriever import HybridRetriever
from app.rag.store import LocalVectorStore


def build(settings: Settings) -> HybridRetriever:
    embeddings = LocalHashEmbeddings(settings.embedding_dim)
    store = LocalVectorStore(settings.index_dir)
    count = ingest(settings, embeddings, store)
    assert count > 0
    return HybridRetriever(settings, embeddings, store)


def test_ingest_and_retrieve_clause(settings: Settings):
    retriever = build(settings)
    hits = retriever.search("一线城市出差住宿费上限是多少？")
    context, citations = retriever.context(hits)
    assert citations
    assert any("报销管理制度#第二条" in item["source"] for item in citations)
    assert "500" in context
    assert context.startswith("<untrusted_doc")


def test_paraphrase_still_hits(settings: Settings):
    retriever = build(settings)
    hits = retriever.search("去上海出差住酒店一晚最多能报多少钱？")
    _, citations = retriever.context(hits)
    assert any("报销管理制度#第二条" in item["source"] for item in citations)


def test_trap_question_returns_no_citation(settings: Settings):
    retriever = build(settings)
    hits = retriever.search("公司是否报销员工的子女教育费用？")
    context, citations = retriever.context(hits)
    assert not citations
    assert "未找到相关规定" in context
