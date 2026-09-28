# M1–M5 阶段总结

本文件记录进入 M6 工程化扩展前的 M1–M5 本地 RAG 基线。项目是学习和实验用的最小 RAG 原型，不是生产系统。

## 已完成内容

- Markdown 与文本型 PDF 文档加载，统一保留来源；PDF 文档保留页码。
- Markdown 标题和段落优先切分；较长内容使用现有滑窗参数。
- 使用 Sentence Transformers 生成 embedding，以 NumPy cosine similarity 排序并返回 Top-K。
- 提供 Markdown 与 PDF Retrieval Evaluation 脚本。
- 通过本地 Ollama `qwen3:4b` 生成答案。
- RAG CLI 将检索结果作为上下文，并显示来源、章节、chunk、分数；PDF 来源可显示页码。

## Markdown Retrieval 基线

```text
文档数：33
Chunks：436
测试问题：50
Top-1 source hit：31/50
Top-3 source hit：36/50
Keyword hit：10/50
Average retrieved score：0.7039
```

该结果是对应语料、测试集、chunk 参数和 embedding 模型的历史快照，不代表通用检索质量。

## PDF Retrieval 基线

```text
测试问题：4
Top-1 source hit：4/4
Top-3 source hit：4/4
Keyword hit：3/4
```

体测标准表格中的“大一大二男生 1000 米跑满分”未召回预期关键词 `1000米` 和 `3'17`。校历题的关键词只检查“国庆”，不能据此认定准确放假日期已验证。

## 模型与运行环境

- Embedding：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Local LLM：Ollama `qwen3:4b`
- Generation API：本机 Ollama `/api/generate`
- 本机检测到：NVIDIA GeForce RTX 5060 Laptop GPU。历史评测没有记录模型实际使用的执行设备，因此不将结果归因于该 GPU。

## 当前限制

- 没有 Vector Database、Reranker、Agent workflow 或 Memory。
- 没有权限控制、文档版本管理、多租户、监控或 Web API。
- PDF 路径以文本提取为主；复杂表格、扫描件和复杂版式处理有限。
- Citation 显示送入上下文的检索来源，不是逐句 claim-level citation。
- 真实工业资料、教师提供的数据、私有文档、PDF、API Key 和本地模型文件不提交到仓库。

## 复现命令

```bash
python src/evaluate_retrieval.py
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_pdf_retrieval.py
python src/rag_demo.py
```

M1–M5 基线已归档；M6 尚未开始。
