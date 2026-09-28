# Retrieval Test Data

`retrieval_test.json` 用于保存 RAG 检索测试案例。

每个测试案例包含以下字段：

- `question`：用户问题。
- `expected_source`：期望召回的文档来源。
- `expected_keywords`：用于人工检查检索结果的关键关键词。
- `answer`：标准答案，人工填写，可选；当前 M1 评测脚本不检查该字段。

从项目根目录运行批量检索评测：

```bash
python src/evaluate_retrieval.py
```

脚本使用当前 M1 检索流程，检查 Top-1/Top-3 来源命中和 Top-3 文本是否包含全部 `expected_keywords`，并输出未命中案例。它不调用 LLM，也不判断答案内容。
