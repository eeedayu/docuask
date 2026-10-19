#!/usr/bin/env python3
"""DocuAsk Streamlit UI: 上传文档 -> 提问 -> 答案 + 引用卡片."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from qa import generate_answer
from retrieve import Retriever

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "data" / "uploads"
COLLECTION = "docuask"

st.set_page_config(page_title="DocuAsk", page_icon="📄", layout="wide")
st.title("DocuAsk — 上传文档，带引用溯源的问答")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(collection=COLLECTION)


with st.sidebar:
    st.header("📤 上传文档")
    files = st.file_uploader("PDF / Markdown", type=["pdf", "md"], accept_multiple_files=True)
    if st.button("开始入库", disabled=not files):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for f in files:
            (UPLOAD_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("切分 + 向量化 + 入库中…"):
            r = subprocess.run(
                [sys.executable, "ingest.py", str(UPLOAD_DIR), "--collection", COLLECTION],
                cwd=BASE, capture_output=True, text=True,
            )
        if r.returncode == 0:
            st.success("入库完成")
            st.cache_resource.clear()
        else:
            st.error("入库失败：\n" + r.stderr[-2000:])

query = st.text_input("💬 提问", placeholder="例如：DocuAsk 用什么模型做向量？")
top_k = st.slider("引用数量", 1, 10, 5)

if st.button("提问", disabled=not query.strip()):
    retriever = get_retriever()
    with st.spinner("检索中…"):
        evidence = retriever.retrieve(query.strip(), top_k=top_k)
    if not evidence:
        st.warning("库里还没有文档，先从左侧上传入库。")
        st.stop()
    with st.spinner("生成答案中…"):
        answer = generate_answer(query.strip(), evidence)
    st.subheader("答案")
    if answer:
        st.markdown(answer)
    else:
        st.info("LLM 未启动，下面是检索到的原始依据。")
    st.subheader("引用")
    for i, e in enumerate(evidence, 1):
        with st.expander(f"[{i}] {e.source} · {e.section or 正文} · {e.score:.3f}"):
            st.markdown(e.text)
