"""Answer generation with local Ollama; degrades gracefully when unavailable."""

from __future__ import annotations

import sys

import ollama

MODEL = "qwen3:4b"
SYSTEM = (
    "你是 DocuAsk 问答助手。只根据下面提供的文档片段回答问题，回答用中文。"
    "如果片段里没有答案，就直说\"文档中没有相关信息\"，不要编造。"
)


def generate_answer(query: str, evidence: list, model: str = MODEL) -> str:
    """Generate a grounded answer. Returns "" when the LLM is unavailable."""
    if not evidence:
        return ""
    ctx = "\n\n".join(
        f"[{i}] (来源:{e.source} / {e.section or 正文})\n{e.text}"
        for i, e in enumerate(evidence, 1)
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"文档片段：\n{ctx}\n\n问题：{query}\n\n请回答，并在关键结论后标注片段编号，如 [1]。"},
            ],
        )
        return resp["message"]["content"]
    except Exception as e:  # noqa: BLE001 - ollama down / model missing
        print(f"[warn] ollama unavailable: {e}", file=sys.stderr)
        return ""
