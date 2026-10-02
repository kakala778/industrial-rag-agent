# M10.1 — Parent Context Controlled Experiment

日期：2026-10-02；代码基准 `8cabb7d`。结论：**ENTER AGENT**。Parent context 有局部排序收益，但当前策略不采用为默认或正式可选 RAG 路径；实现保留为 offline experiment。

## 1. Motivation

M9.2 审计发现 Q25 的完整 table row 仍存在于 MinerU；完整行 child 在 RRF union rank23，被固定 Hybrid20 截掉；进入候选的相邻 child 从标识符内部开始，仍被旧 matcher 接受。目标是验证恢复已有上下文后再 BGE 排序的机制，不实现全量 contextual retrieval。

本轮保持原 PDF/cache、structured、chunking、embedding、BM25/tokenizer、RRF、Top20、BGE、Top3、QA/GT/matcher 不变。唯一主要变量为 BGE 的输入文本。没有生成 LLM prefix、重新 embedding、扩大候选或调用新模型。

## 2. Parent Mapping

映射键为 source/page/block_type/block_index；chunk_id 用于 child provenance。先要求唯一 parent Document，再要求 child 原文在 parent 中唯一出现。重复 metadata、重复子串、缺父块等情况均回退 child 并记录状态，不猜测。

Q25 初始验证：一个 parent、一个 exact child occurrence；parent 1,226 字符、child 500 字符。child 相交八个序列化行，其中起止边界行被截断。按只读 parent 匹配可以恢复完整边界行。

这不是永久 ID 系统：内部 provenance 保存完整结构键，公开结果使用匿名案例与 corpus ordinals。文本和真实来源标识只留 ignored 输出。

## 3. Context Recovery Strategy

实现：[parent_context.py](../src/parent_context.py)。它不读取 QA、GT 或 evaluator 标签。

- Table：恢复 child 截断的边界完整行；保留表格前三个序列化行作为 leading context；完整 interior rows 已在 child 中，不再全部重复。
- Text/image/chart 等其他 block：只使用同 parent 的前后各最多 120 字符 neighborhood，不跨页拼接。
- 总 query + text + special tokens 上限固定为 768，并取实际 tokenizer/model 有限长度限制的更小值。没有多轮长度调参。
- parent 增强超预算时原样回退 child；child 本身超预算时停止，防止悄悄改变 frozen baseline。

Table leading context **不自动等同语义表头**。能保留哪些列关系，需要针对原 PDF 和实际 serialized rows 人工复核；helper 没有推断 colspan/rowspan、创建新的 cell 结构或补写字段。

格式使用 `[Table leading context]` / `[Complete boundary rows]` / `[Retrieved child]`，或 `[Parent neighborhood]` / `[Retrieved child]`。无 answer、QA category、范围标签注入。

## 4. Experimental Arms & Reproduction

| Arm | Recall | BGE input | Evaluated / returned candidate |
|---|---|---|---|
| A | 原 Hybrid20 | 原 child text | 原 child |
| B | 相同 Hybrid20 | child + bounded existing context | 原 child，仅排序分数改变 |

Runner：[run_m10_parent_context_experiment.py](../evaluation/run_m10_parent_context_experiment.py)。运行命令（在应用目录）：

```powershell
python evaluation/run_m10_parent_context_experiment.py
```

使用已冻结 M9.1/M9.2 summary 的 candidate ordinals，检查 PDF/QA/cache inventory、独立集 manifest、corpus 数量、结构身份、原 evidence ranks，恢复同一 pool。A 每题完整 BGE 排序必须复现历史排序，才允许比较 B。

本轮不执行 embedding、BM25 rebuild 或 MinerU。原型工具的 generation/citation 文件在 M9.3 已合法改变；旧 generation helper hash 差异被单列，未忽略 retrieval/parser/reranker/matcher 差异。当前核心代码和输入仍进行运行前后 hash 检查。

模型只从本地 cache 加载；环境设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。详细 parent/child 文本和 provenance 写入 `outputs/m10_parent_context/`，不进入 Git。

## 5. Q25 Mechanism Test

Q25 先于两个 cohort 执行。停止门槛要求目标 child 确实得到 boundary-row expansion 且其 BGE rank 改善；竞争候选变化使 child-only fallback 升位，不算机制成立。

本轮由助手逐例查看本地原 PDF 页图和 structured row 做视觉核对，不是新增独立人工盲审：目标行索引为 10，有七个序列化列，原 child 缺完整标识符。恢复行在同一 parent 中包含完整标识符、计量单位和计算规则；与原 PDF 同行对应。这里的 value 是计算规则字段，不能只按数值 keyword 命中判断。

输入关联与返回候选关联必须分开：BGE 看到恢复行，并不意味着最终返回 child 已含完整标识符。最终 child 文本没有被替换，正式评价仍使用旧 child matcher。

| Q25 target child | Rank / length |
|---|---:|
| Frozen Hybrid rank | 14 |
| A BGE rank | 7 |
| B BGE rank | 1 |
| A pair tokens | 394 |
| B pair tokens | 678 |

恢复的是同 table parent 中的完整边界行 10 与 17，不是扩大 pool 把原 union rank23 候选重新加入。目标行 10 的 identifier—规则—unit 关联经 PDF/structured output 对照确认；leading context 中的对应列标题也经该案例的页面核对确认。helper 本身不具备通用 header 识别能力。

保存结果中的目标状态为 EXPANDED / table_boundary_rows。审查补强了“目标自身确实恢复”的 gate；保存的 Q25 输出经更严格 gate 再次核验通过，没有改 BGE 输入或补跑一套参数。

## 6. Controls

- Q06 与冻结集内已成功 table cases：检验已有 success 是否回退、邻行稀释与 header/context 竞争。
- 普通 text success：检验同 parent neighborhood 是否引入负面变化。
- Q19/Q24：scope ambiguity controls。查询仍不指定全部范围；即使指标升位，也不能宣称 parent context 恢复用户未表达的意图。
- 独立 I17：正确候选的 parent 唯一，child 从序列化行 4 内开始；目标行 6 已完整。进一步核对 PDF 与输入，发现 column header 在 A child 中缺失、在 B leading context 中存在。它属于相关的 table context 缺失，但不是 Q25 的目标标识符截断机制；不能将 rank gain 的内因仅归于 header，因为其他候选也同时增强。

| Case | Role | A evidence rank | B evidence rank | Observation |
|---|---|---:|---:|---|
| Q02 / Q03 | Table success | 1 / 1 | 1 / 1 | 保持成功 |
| Q04 | Table regression | 1 | 4 | 退出 evidence Top3；原 child 已含 header，补边界行后仍回退 |
| Q06 | Required table control | 1 | 1 | 保持成功；child 原本就是完整 parent |
| Q11 / Q18 | Table success | 1 / 1 | 1 / 1 | 保持成功 |
| Q01 / Q05 / Q15 | Text success | 1 / 2 / 1 | 1 / 2 / 1 | 保持成功 |
| Q19 | Ambiguity control | 5 | 5 | 仍失败且 scope 不明确 |
| Q24 | Ambiguity control | 13 | 11 | 未进入 Top3；仍缺文档范围 |
| I17 | Independent diagnostic gain | 4 | 1 | 目标行原本完整；新增表头/非目标边界行上下文 |
| I02 / I03 / I16 / I20 / I21 | Independent table success | 1 | 1 | 各自保持成功 |

Q19/Q24 的新输入只取对应候选的真实 parent，不附加 expected source/page、答案或 evaluator 标签。没有将它们重新定义为“已消除 query ambiguity”。

## 7. Paired Results

使用 M9.1 的 32 scored QA 与 M9.2 的 21 independent QA。M9.1 原 dataset 共 35 行，另外三题 unanswerable 没有候选，不进入 retrieval 指标分母。历史 Q17 的 GT uncertainty 敏感性分析单列，不套用到 I17。

正式 paired run 完成；两个 cohort 的 A 完整排序及最终 baseline 指标均复现历史结果。

| Dataset | Mode | Top1 page | Top3 page | Normalized evidence | Strict/raw evidence |
|---|---|---:|---:|---:|---:|
| Original 32 | A Hybrid+BGE child | 21/32 | 23/32 | 17/32 | 13/32 |
| Original 32 | B Hybrid+ParentContext+BGE | 20/32 | 24/32 | 17/32 | 13/32 |
| Independent 21 | A Hybrid+BGE child | 16/21 | 17/21 | 15/21 | 14/21 |
| Independent 21 | B Hybrid+ParentContext+BGE | 14/21 | 18/21 | 16/21 | 15/21 |

candidate identity 未变，原 Recall@20 为 21/32 与 16/21，不将 reranking gains 写成 recall gains。

| Dataset | Metric | Gains | Regressions | Unchanged |
|---|---|---:|---:|---:|
| Original 32 | Top1 page | 1 | 2 | 29 |
| Original 32 | Top3 page | 1 | 0 | 31 |
| Original 32 | Normalized evidence | 1 | 1 | 30 |
| Original 32 | Strict/raw evidence | 1 | 1 | 30 |
| Independent 21 | Top1 page | 1 | 3 | 17 |
| Independent 21 | Top3 page | 1 | 0 | 20 |
| Independent 21 | Normalized evidence | 1 | 0 | 20 |
| Independent 21 | Strict/raw evidence | 1 | 0 | 20 |

历史 Q17 排除后的 31 题 sensitivity：A/B Top1 page 均 20/31；Top3 page 22/31 → 23/31；normalized 均 17/31；strict 均 13/31。主表仍保持原 32 分母与 GT 不变。

## 8. Row Association Review

诊断区分 COMPLETE_ASSOCIATION / PARTIAL / INCORRECT / NOT_APPLICABLE，独立于旧 matcher。完整关系要求同一真实 row 的 identifier、字段值/规则、unit 与可验证表头关系，不接受来自不同 row 的词汇拼接。

本轮 row recovery 为真实边界行原文，不代表所有 table cases 都有人工关系评分。未复核案例标记未评审，不用字符串匹配冒充人工正确率。

| Case | A BGE target-row input | B BGE target-row input | Returned child A / B | Review |
|---|---|---|---|---|
| Q25 | PARTIAL | COMPLETE_ASSOCIATION | PARTIAL / PARTIAL | PDF 与 structured row 确认完整标识符、规则、unit、对应列标题 |
| I17 | COMPLETE_ASSOCIATION，header 缺失 | COMPLETE_ASSOCIATION，header 存在 | COMPLETE_ASSOCIATION / COMPLETE_ASSOCIATION | 目标行本来完整；PDF 确认该行名称/数量/unit 的对应 |
| Q06 | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE / NOT_APPLICABLE | 不是本轮 identifier-value-unit 恢复问题；页面有合并单元格，rank control 保持 |

这里 COMPLETE_ASSOCIATION 指被核对的目标行，不表示整个 child 所有行或整张表完整。Q06 的完整 parent 已在 child 中，不能将重复 leading context 说成新增表格事实。

## 9. Regressions

正式回退依据 baseline success → B failure，page / normalized / strict 分开计数。正文只记录匿名 ID。Q19/Q24 即使 formal hit 改变，scope ambiguity 仍单独保留。

- Evidence/strict gain：Q25、I17。Evidence/strict regression：Q04。
- Top3 page gain：Q25、I17；无 Top3 page regression。
- Top1 page gain：Q25、I17；regressions：Q04、Q17、I08、I09、I14。历史 Q17 保留 uncertainty 标记；排除后仍有其他 Top1 回退。
- Q04 正确 child 从 1→4，但 Top3 仍能命中目标页。这说明 page hit 增加不等于字段 evidence 都保持成功。

Q04 原 child 500 字符、parent 680 字符；A 已含表头。B 为该 child 恢复边界行 6，pair tokens 406→676，rank 1→4。原始资料、目标与 candidate identity 均未变。这是当前增强策略的真实 ranking regression，不能解释成修复了 OCR，也不能仅凭输入观察断言 BGE 内部原因。

## 10. Runtime / Token Cost

记录每个 query-candidate pair 的实际 tokenizer 长度、A/B BGE runtime、context expansion runtime。CPU 单轮描述不是生产 benchmark；不将候选重放省下的 embedding/index 成本记成 parent context 的性能收益。

| Cohort | Candidate pairs | A mean pair tokens | B mean pair tokens | A mean BGE s/query | B mean BGE s/query | Expansion s/query |
|---|---:|---:|---:|---:|---:|---:|
| Original 32 | 640 | 274.57 | 407.56 | 23.29 | 41.08 | 0.0363 |
| Independent 21 | 420 | 248.05 | 370.22 | 13.84 | 22.99 | 0.0232 |
| All scored 53 | 1,060 | 264.06 | 392.77 | 19.54 | 33.91 | 0.0311 |

全体平均输入 tokens +48.7%；平均 BGE 时间 +14.37s/query（+73.5%）。A 最大 pair 为 462 tokens，B 最大为 768；token 均含 query/special tokens。BGE load 为 24.03s，另计。

817/1,060 pairs 被增强；243 因 context 超预算保持 child-only。无 missing/ambiguous parent/occurrence fallback。context 类型为 table_boundary_rows 231、table_prefix 131、text_neighborhood 455、child_only 243。

另有 405/1,060 pairs 的 child 原本已等于完整 parent；格式或重复文本会增加输入，但没有新增事实。此策略没有单独隔离“新上下文”与“重复/格式”的效果。这是解释收益和回退的限制，本轮没有增加第三套 policy 调优。

CPU、A 后 B 的单次执行、不同 cohort 的输入分布与系统负载都影响 wall time；不是随机重复 latency benchmark。

## 11. Decision & Agent Readiness Update

**不采用当前策略为 default、正式 optional 或 table-only application path；保留 experimental only。最终决定：ENTER AGENT。**

Q25 的机制成立，I17 也有一次独立 gain，但 original evidence 无净提升、有已成功 table case 回退、两组 Top1 page 下降；token/latency 成本明显增加。table-only 也不能由此证明安全，因为 Q04 同样是 table。一个独立 gain 不足以满足“若干独立 case 改善、几乎无 regression”的稳定可选集成标准。

本轮既不否定所有 parent retrieval，也不为证明其价值继续改 cap、RRF、Top20、模型或 matcher。得到的边界是：已有上下文能帮助某些排序，但不能可靠地全面开启；仅供 BGE 的完整行不自动进入返回/生成 context。

本轮作为 Agent 前最后一轮 RAG 实验收尾，不追加新的 RAG milestone，也不在本轮实现 Agent。

Agent 的 search/lookup 契约、原始 parent 的有界 lookup、scope clarification 与答案 grounding 验收仍需完成。仅供 reranker 的上下文增强不代替这些能力。

## 12. Verification

验证 synthetic tests 覆盖唯一/缺失/重复 parent、重复 occurrence、row/header prefix 保留、token cap、provenance、不改 child identity、候选 replay 校验、输出路径和停止门槛。历史模型 QA 不进入测试文件。

运行前后 protected inputs/core 校验通过；MinerU calls=0、embedding calls=0；原候选数量、身份和 citation provenance 保持。BGE A 全排序复现后才记录 B，对保存的 Q25 strict gate 再核验通过。

匿名结果及本地详细 context 在 `outputs/m10_parent_context/run_20261002T142752Z/`，Git ignored。公开文件不包含真实 PDF 文件名、QA/答案、工业正文或 parent text。下一次复现需要相同本地 frozen QA/cache/M9 summaries 和 cached BGE；缺失输入会停止，不合成替代。

独立代码审查发现并补强 target-expansion gate，未发现剩余关键问题。`python -m compileall src evaluation` 通过；`python -m unittest discover -s tests` 143 tests 通过。Git diff 与未跟踪新文本的空白/链接检查、private artifacts/content 检查另行完成；测试通过不代表答案生成准确率。

## 13. Explicit Answers

| Question | Answer |
|---|---|
| 1. Q25 能唯一映射吗？ | 能：唯一 table parent + 唯一 child occurrence |
| 2. 恢复什么？ | 被截断的完整边界行 10/17 + 有限 table leading context；不是召回新 candidate |
| 3. Q25 ranks？ | Hybrid14，A BGE7 → B BGE1 |
| 4. 完整关联恢复了吗？ | BGE 输入恢复；返回 child 仍 PARTIAL，没有声称生成端已修复 |
| 5. Q06 等成功例回归吗？ | Q06 不回退；Q04 evidence 1→4，真实回退 |
| 6. Q19/Q24 ambiguity？ | 保留；5→5、13→11，均不进 Top3，不补猜用户 scope |
| 7. I17 同类吗？ | 有相关 header context 缺失，但目标行原本完整；不是同一 target-row 截断机制；4→1 |
| 8. 原 32 总体？ | Top3 page 23→24；normalized17、strict13 不变；Top1 page21→20 |
| 9. 独立 21 总体？ | Top3 page17→18；normalized15→16；strict14→15；Top1 page16→14 |
| 10. Gains/regressions？ | normalized：原集1/1，独立1/0；page/strict/unchanged 分列于第7节 |
| 11. 成本？ | mean tokens264.06→392.77；mean BGE19.54→33.91s；expansion约31ms/query |
| 12. 值得采用吗？ | 当前 broad policy 不值得正式采用；局部机制证据保留 |
| 13. 采用级别？ | experimental only，未接默认/optional/table-only RAG |
| 14. 不采用原因？ | baseline table 回退、原集无净 evidence gain、Top1 下降、成本增加、样本与机制证据有限 |
| 15. 是否进入 Agent？ | **ENTER AGENT**；后续先做有界 search/lookup/state/grounding，不再加前置 RAG 实验 |
