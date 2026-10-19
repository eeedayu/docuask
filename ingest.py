#!/usr/bin/env python3
"""DocuAsk ingest pipeline.

PDF / Markdown -> chunk (heading-first, then length with overlap)
-> BGE-M3 dense embeddings -> Qdrant.

Usage:
    python ingest.py [DATA_DIR] [--collection docuask] [--recreate]

Idempotent: point ids are deterministic hashes of (source, chunk_index),
so re-running upserts instead of duplicating.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # characters of overlap
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(t.strip() for t in pages if t.strip())


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) sections on Markdown headings."""
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if cur_lines or cur_heading:
                sections.append((cur_heading, "\n".join(cur_lines).strip()))
            cur_heading = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines or cur_heading:
        sections.append((cur_heading, "\n".join(cur_lines).strip()))
    sections = [(h, b) for h, b in sections if b]
    return sections or [("", text.strip())]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy character chunking with overlap, preferring newline boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start + size // 2:
                end = nl
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_documents(data_dir: Path):
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".md":
            yield path, read_markdown(path)
        elif suffix == ".pdf":
            try:
                yield path, read_pdf(path)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] skip {path.name}: {e}")


def build_chunks(text: str) -> list[tuple[str, str]]:
    """Return (section_heading, chunk_text) pairs."""
    out: list[tuple[str, str]] = []
    for heading, body in split_by_headings(text):
        for c in chunk_text(body):
            out.append((heading, c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDF/Markdown docs into Qdrant")
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--collection", default="docuask")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    client = QdrantClient(url=args.qdrant_url)
    if args.recreate and client.collection_exists(args.collection):
        client.delete_collection(args.collection)
        print("[qdrant] collection dropped")
    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[qdrant] collection {args.collection} (re)created")

    print(f"[model] loading {EMBED_MODEL} ...", flush=True)
    model = SentenceTransformer(EMBED_MODEL)

    total = 0
    for path, text in iter_documents(data_dir):
        chunks = build_chunks(text)
        if not chunks:
            print(f"[skip] {path.name}: no text")
            continue
        embeddings = model.encode(
            [c for _, c in chunks],
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        points = []
        for i, ((heading, chunk), vec) in enumerate(zip(chunks, embeddings)):
            pid = int(hashlib.sha256(f"{path.name}#{i}".encode()).hexdigest()[:16], 16)
            points.append({
                "id": pid,
                "vector": vec.tolist(),
                "payload": {
                    "source": path.name,
                    "chunk_index": i,
                    "section": heading,
                    "text": chunk,
                },
            })
        client.upsert(collection_name=args.collection, points=points)
        total += len(points)
        print(f"[ok] {path.name}: {len(points)} chunks")
    print(f"done: {total} chunks -> {args.collection}")


if __name__ == "__main__":
    main()
