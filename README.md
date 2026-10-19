<div align="center">

# DocuAsk

**Upload documents. Ask questions. Get answers with citations.**

**上传文档，提问，得到带引用溯源的答案。**

</div>

---

## What it does / 项目简介

DocuAsk is a retrieval-augmented Q&A demo: upload a PDF or Markdown document, ask questions in natural language, and get a grounded answer with **clickable source citations** — every key claim points back to the exact document chunk it came from.

DocuAsk 是一个带引用溯源的文档问答 demo：上传 PDF/Markdown 文档，用自然语言提问，得到有**依据的答案**——每个关键结论都标注了来源片段编号，可回溯原文。

**Live demo:** https://docuask.eeedayu.dpdns.org/

## How it works / 原理

```
PDF/MD ──▶ chunking ──▶ BGE-M3 embeddings ──▶ Qdrant (1024-dim, cosine)
                                                      │
User question ──▶ hybrid retrieval ──────────────────┘
                   (dense + BM25 → RRF fusion → bge-reranker-v2-m3)
                                                      │
                                              top-k evidence
                                                      ▼
                                          DeepSeek generates answer
                                          with [1][2]... citations
```

1. **Ingest** (`ingest.py`) — PDF/Markdown chunking → BGE-M3 → Qdrant `docuask` collection. Idempotent: re-ingesting the same doc doesn't duplicate chunks.
2. **Retrieve** (`retrieve.py`) — hybrid search: dense vectors + BM25 keyword search, fused with RRF, then reranked with `bge-reranker-v2-m3`. Returns structured evidence with source/section metadata.
3. **Answer** (`qa.py` + `app.py`) — DeepSeek generates a grounded answer from the evidence; the UI renders it with citation cards. If the model is unreachable, the app degrades to retrieval-only instead of failing.

## Evaluation / 评测

20-question eval set (`eval/`):

| Metric | Score |
|---|---|
| recall@1 | 85% |
| recall@3 / recall@5 | 100% |
| Answer hit rate | 90% |

## Tech Stack / 技术栈

| Layer | Choice |
|---|---|
| Embeddings | BGE-M3 (dense + sparse) |
| Vector DB | Qdrant (1024-dim, cosine) |
| Reranker | bge-reranker-v2-m3 |
| Answer generation | DeepSeek (`deepseek-chat`) |
| UI | Streamlit |
| Language | Python |

## Quickstart / 快速开始

```bash
pip install -r requirements.txt

# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Ingest documents (put PDFs/Markdown in data/, or pass a directory)
python ingest.py
# python ingest.py /path/to/your/docs

# 3. Ask (CLI)
python -c "
from retrieve import search
from qa import generate_answer
ev = search('你的问题')
print(generate_answer('你的问题', ev))
"

# 4. Or launch the web UI
DEEPSEEK_API_KEY=sk-... streamlit run app.py
```

## Project Structure / 项目结构

```
├── app.py           # Streamlit UI: upload → ingest → ask → answer + citations
├── ingest.py        # PDF/MD chunking → BGE-M3 → Qdrant
├── retrieve.py      # hybrid dense+BM25 → RRF → rerank
├── qa.py            # grounded answer generation (DeepSeek)
├── eval/            # 20-question eval set + results
│   ├── corpus/      # sample docs
│   ├── questions.jsonl
│   └── results.md
└── requirements.txt
```

## License

MIT
