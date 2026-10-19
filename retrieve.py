#!/usr/bin/env python3
"""DocuAsk retrieval chain: hybrid (dense + BM25) -> RRF -> BGE rerank.

Returns structured evidence for answer generation:
    [{"text": ..., "source": ..., "section": ..., "chunk_index": ..., "score": ...}]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60


@dataclass
class Evidence:
    text: str
    source: str
    section: str
    chunk_index: int
    score: float


def tokenize_zh(text: str) -> list[str]:
    # char-level for Chinese + alnum runs; good enough for demo BM25
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())


class Retriever:
    def __init__(self, qdrant_url: str = "http://127.0.0.1:6333",
                 collection: str = "docuask", use_reranker: bool = True):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        print("[model] loading", EMBED_MODEL, flush=True, file=sys.stderr)
        self.embed = SentenceTransformer(EMBED_MODEL)
        self.corpus: list[dict] = []
        self._load_corpus()
        self.bm25 = BM25Okapi([tokenize_zh(d["text"]) for d in self.corpus]) if self.corpus else None
        self.reranker = None
        if use_reranker:
            print("[model] loading", RERANK_MODEL, flush=True, file=sys.stderr)
            self.reranker = CrossEncoder(RERANK_MODEL)

    def _load_corpus(self) -> None:
        if not self.client.collection_exists(self.collection):
            return
        points, _ = self.client.scroll(self.collection, limit=10000, with_vectors=False)
        for p in points:
            pl = p.payload or {}
            self.corpus.append({
                "id": p.id,
                "text": pl.get("text", ""),
                "source": pl.get("source", ""),
                "section": pl.get("section", ""),
                "chunk_index": pl.get("chunk_index", 0),
            })

    def _dense(self, query: str, top_k: int) -> list[tuple]:
        qv = self.embed.encode([query], normalize_embeddings=True)[0].tolist()
        hits = self.client.query_points(
            collection_name=self.collection, query=qv, limit=top_k).points
        return [(h.id, h.score) for h in hits]

    def _sparse(self, query: str, top_k: int) -> list[tuple]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_zh(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.corpus[i]["id"], float(scores[i])) for i in ranked if scores[i] > 0]

    @staticmethod
    def _rrf(runs: list[list[tuple]], k: int = RRF_K) -> list[tuple]:
        fused: dict = {}
        for run in runs:
            for rank, (pid, _) in enumerate(run):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 5, fusion_k: int = 20) -> list[Evidence]:
        if not self.corpus:
            return []
        fused = self._rrf([self._dense(query, DENSE_TOP_K),
                           self._sparse(query, BM25_TOP_K)])[:fusion_k]
        by_id = {d["id"]: d for d in self.corpus}
        cands = [(pid, by_id[pid]) for pid, _ in fused if pid in by_id]
        if self.reranker and cands:
            pairs = [(query, d["text"]) for _, d in cands]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)
            cands = [c for c, _ in ranked]
            final_scores = [float(s) for _, s in ranked]
        else:
            score_of = dict(fused)
            final_scores = [score_of[pid] for pid, _ in cands]
        return [Evidence(text=d["text"], source=d["source"], section=d["section"],
                         chunk_index=d["chunk_index"], score=s)
                for (_, d), s in zip(cands[:top_k], final_scores[:top_k])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval over the docuask collection")
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--collection", default="docuask")
    args = ap.parse_args()
    r = Retriever(collection=args.collection, use_reranker=not args.no_rerank)
    for e in r.retrieve(args.query, top_k=args.top_k):
        print(json.dumps(asdict(e), ensure_ascii=False))


if __name__ == "__main__":
    main()
