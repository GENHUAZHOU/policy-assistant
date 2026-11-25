"""回归评测：python -m evals.run_eval（CI 中同样调用）。"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.config import ROOT, get_settings
from app.models import build_embeddings
from app.rag.ingest import ingest
from app.rag.retriever import create_retriever
from app.security import ToolContext, ToolPermissionError
from app.tools import invoke_tool

CASES_PATH = Path(__file__).with_name("cases.yaml")


def main() -> int:
    settings = get_settings()
    retriever = create_retriever(settings)
    if retriever.store.count() == 0:
        ingest(settings, build_embeddings(settings), retriever.store)

    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    passed = 0
    failures: list[str] = []

    for case in cases:
        try:
            if case["type"] == "injection":
                try:
                    invoke_tool(
                        case["tool"],
                        case["args"],
                        ToolContext(thread_id=f"eval-{case['id']}", role=case["role"]),
                        settings=settings,
                    )
                except ToolPermissionError:
                    pass
                else:
                    raise AssertionError("注入攻击未被权限矩阵拦截")
            else:
                hits = retriever.search(case["question"])
                context, citations = retriever.context(hits)
                if case.get("expect_no_hit"):
                    assert not hits, f"陷阱问题不应命中条款，实际命中 {[item['source'] for item in hits]}"
                else:
                    assert citations, "未检索到任何条款"
                    sources = [citation["source"] for citation in citations]
                    assert any(case["expect_citation"] in source for source in sources), (
                        f"引用错误：期望 {case['expect_citation']}，实际 {sources}"
                    )
                    for keyword in case.get("expect_keywords", []):
                        assert keyword in context, f"上下文缺少关键词：{keyword}"
            passed += 1
            print(f"[PASS] {case['id']} {case['type']}")
        except Exception as exc:
            failures.append(f"{case['id']}: {exc}")
            print(f"[FAIL] {case['id']} {case['type']} -> {exc}")

    print(f"\n结果：{passed}/{len(cases)} 通过")
    if failures:
        print("失败用例：\n  - " + "\n  - ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
