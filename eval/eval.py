#!/usr/bin/env python3
"""DocuAsk eval: 20 题评测集 -> recall@k + 答案关键词命中率 -> eval/results.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa import generate_answer
from retrieve import Retriever

BASE = Path(__file__).parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="docuask_eval")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--skip-gen", action="store_true", help="只测检索，跳过答案生成")
    args = ap.parse_args()

    questions = load_questions(BASE / "questions.jsonl")
    r = Retriever(collection=args.collection)
    rows = []
    for i, q in enumerate(questions, 1):
        ev = r.retrieve(q["question"], top_k=args.top_k)
        sections = [e.section for e in ev]
        exp = q["expected_section"]
        r1, r3, r5 = exp in sections[:1], exp in sections[:3], exp in sections[:5]
        hit = None
        if not args.skip_gen and ev:
            ans = generate_answer(q["question"], ev[:3])
            hit = any(kw.lower() in ans.lower() for kw in q["keywords"])
        rows.append({"q": q["question"], "expected": exp, "got": sections[:3],
                     "r@1": r1, "r@3": r3, "r@5": r5, "hit": hit})
        print(f"[{i}/{len(questions)}] r@1={int(r1)} r@3={int(r3)} r@5={int(r5)} "
              f"hit={hit} :: {q["question"][:24]}", flush=True)

    n = len(rows)

    def avg(key: str) -> float:
        vals = [x[key] for x in rows if x[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {"n": n, "recall@1": avg("r@1"), "recall@3": avg("r@3"),
               "recall@5": avg("r@5"), "answer_hit": avg("hit")}

    out = ["# DocuAsk 评测结果", "",
           f"题量：{n}；集合：{args.collection}；top_k={args.top_k}；评测语料见 eval/corpus/", "",
           "| 指标 | 得分 |", "|---|---|"]
    for k, v in summary.items():
        if k != "n":
            out.append(f"| {k} | {v:.1%} |")
    out += ["", "## 明细", "",
            "| # | 问题 | 期望章节 | 命中章节(top3) | r@1 | r@3 | r@5 | 答案命中 |",
            "|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(rows, 1):
        got = "、".join(row["got"]) or "-"
        h = "-" if row["hit"] is None else int(row["hit"])
        out.append(f"| {i} | {row["q"]} | {row["expected"]} | {got} | "
                   f"{int(row["r@1"])} | {int(row["r@3"])} | {int(row["r@5"])} | {h} |")
    (BASE / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False))
    print("written to eval/results.md")


if __name__ == "__main__":
    main()
