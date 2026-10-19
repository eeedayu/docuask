# DocuAsk 部署运维

## 环境要求
Python 3.14 venv。torch 必须先装 CPU 版以保持精简，再装 requirements.txt。其余依赖：qdrant-client、sentence-transformers、pypdf、rank-bm25、streamlit、ollama。

## 启动步骤
1. 建 venv 并安装依赖（torch 先用 CPU index）。
2. 启动 ollama serve 并执行 ollama pull qwen3:4b。
3. 运行 python ingest.py data/ 把文档入库。
4. 运行 streamlit run app.py，浏览器打开 8501 端口。

## 上传文档
侧栏上传的 PDF/MD 会保存到 data/uploads/ 目录，该目录被 .gitignore 排除，不会入库。入库走 ingest.py 的同一套切分与向量化流程。

## 重建集合
ingest.py 加 --recreate 参数会先删除旧集合再重建。不加该参数时重跑入库是幂等的，不会产生重复向量。

## 常见问题
Ollama 连不上先检查 ollama serve 是否在跑，默认端口 11434。VPS 无 GPU，Ollama 与模型推理均为 CPU 模式，4b 模型问答约几十秒一条，可接受。Qdrant 确认 docker 容器在运行且只监听本机。
