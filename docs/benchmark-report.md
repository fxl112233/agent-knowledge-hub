# MultiHop-RAG 评测报告

## 结论先行

项目已经完成一轮可复现的检索与回答实验，但结果不支持“GraphRAG 让回答准确率大幅提升”这一宣传式结论。最稳定、也最适合写入简历的发现是：在固定 100 题配对实验中，Hybrid 将 All-evidence Hit@10 从 `45.33%` 提升至 `50.67%`，增加 `5.34` 个百分点；Evidence Recall@10 从 `74.44%` 提升至 `76.78%`。

同一实验里，Hybrid 严格 Answer EM 为 `57.33%`，低于 Vector 的 `60.00%`，配对差值 `-2.67` 个百分点，95% bootstrap CI 为 `[-10.67, 5.33]` 个百分点。因此不能声称图检索显著提升了最终回答准确率。最终默认配置关闭造成该退化的实验性通用 ComparisonTool，保留经过验证的 RRF、两级精排、文档内二次检索与时间语义工具。

## 实验设置

- 数据集：MultiHop-RAG，公开语料经过确定性准备后使用 100 篇文档子集。
- 开发门控：seed `42` 固定分层 100 题，其中比较、推断、时间、无答案各 25 题。
- 对照：Vector 与 Hybrid 共用同一 QueryPlan、BGE Reranker、二次检索、Top-K 和答案模型；唯一核心变量是是否加入 Neo4j 图候选。
- 检索深度：正式检索指标使用 Top-10。每题标准证据通常是 2–4 篇，而不是 10 篇；Top-10 表示系统最多返回 10 个候选文档/证据单元。
- Reranker：`BAAI/bge-reranker-v2-m3`，初始候选 40，最多选 10 篇文档，每文档最终最多 2 个 chunk。
- 生成：temperature `0`；所有答案引用必须映射到真实 `chunk_id`。
- 置信区间：配对 bootstrap 2,000 次，seed `42`。

## 指标定义

| 指标 | 计算方式 | 回答的问题 |
|---|---|---|
| Evidence Recall@10 | Top-10 命中的标准证据数 / 标准证据总数 | 应找的证据找回了多少 |
| All-evidence Hit@10 | 一题全部标准证据都进入 Top-10 记 1，否则 0 | 多跳证据是否完整 |
| Document Precision@10 | Top-10 中标准证据数 / 返回数 | 候选噪声有多大 |
| Answer EM | 规范化预测与标准答案完全一致 | 严格回答准确率 |
| Token F1 | 预测与标准答案的 token 级 F1 | 部分匹配程度 |
| Citation Recall | 引用覆盖的标准证据比例 | 答案是否引用对来源 |
| Rejection F1 | 把无答案题视为正类计算 F1 | 是否会在证据不足时拒答 |

## 100 题配对结果（v5 消融）

| 指标 | Vector | Hybrid | Hybrid - Vector |
|---|---:|---:|---:|
| Evidence Recall@10 | 74.44% | 76.78% | +2.33 pp |
| All-evidence Hit@10 | 45.33% | 50.67% | +5.34 pp |
| Document Precision@10 | — | 32.45% | — |
| Answer EM | 60.00% | 57.33% | -2.67 pp |
| Citation Recall | 40.67% | 43.78% | +3.11 pp |
| Rejection F1 | — | 65.75% | — |
| 完整回答 p95 | — | 29.68 s | — |

Hybrid 分题型 Answer EM：比较题 `36%`、推断题 `92%`、时间题 `44%`。主要瓶颈不是推断题，而是 Yes/No 比较题的证据绑定与拒答边界。

## 版本消融

| 版本 | 核心改动 | Hybrid Recall@10 | All-evidence Hit@10 | Hybrid EM | 相对 Vector EM |
|---|---|---:|---:|---:|---:|
| v3 | Reranker + 文档内二次检索 + 时间工具 | 67.11% | 37.33% | 50.67% | -4.00 pp |
| v4 | 槽位级 RRF + 时间证据补检 | 76.58% | 45.95% | 59.46% | +1.35 pp |
| v5 | 引用绑定 + 自适应图融合 + 通用比较工具 | 76.78% | 50.67% | 57.33% | -2.67 pp |

v4 因一次查询规划 API 错误实际完成 99 题，不能与完整 100 题做严格等量比较；它只用于判断策略方向。v5 完成 200/200 条 Vector/Hybrid 记录，业务错误和 Rerank fallback 均为 0。

## 原因分析与最终取舍

1. 图候选提高了跨文档证据完整性，但答案模型不一定能把更多证据转化为更高 EM。更多候选也可能带来近似事实和时间版本噪声。
2. v5 的通用 ComparisonTool 对缺槽位采取保守拒答，50 道 Hybrid 比较路由题中有 23 道返回不可回答，其中 7 道实际已经完整召回证据，直接损失 EM。
3. MultiHop-RAG 的比较/时间题常要求输出严格 `Yes/No`。答案语义正确但格式不一致仍会被 EM 判错，因此需同时看 EM、Token F1、引用与拒答指标。
4. p95 约 30 秒主要受外部模型端点排队与生成延迟影响，当前单机 Compose 不满足生产级 2 秒 SLA，简历不应写“实时低延迟”。

最终默认策略采用 v4 的语义时间路由，并保留 v5 已验证的引用绑定、自适应图融合、错误重试和上下文压缩；`COMPARISON_TOOL_ENABLED=false`。这一组合已通过独立固定 10 题、Vector/Hybrid 共 20 条记录的工程契约回归：20/20 完成、0 业务错误、0 Rerank fallback，结构化响应与引用映射全部通过。但 10 题不用于估计质量，在重新跑完固定 100 题前，不把它宣称为新的量化最优版本。

## 可写与不可写

可以写：

- “在固定 100 题配对实验中，Hybrid 的 All-evidence Hit@10 从 45.33% 提升至 50.67%。”
- “建立 Vector/Hybrid 配对评测、分层采样、断点续跑和 bootstrap 95% CI。”
- “实验发现检索增益未稳定传导到答案 EM，并据此回滚收益为负的比较工具。”

不能写：

- “GraphRAG 让回答 F1 提升 22%”。
- “整体准确率从 78% 提升到 94%”。
- “支持千级文档实时问答”或“p95 低于 2 秒”。
- “CDC 效率提升 95%”，除非后续完成固定事件集的全量重建对照。

## 可复现产物

- v3/v4/v5 原始 JSONL、QueryPlan 缓存、summary 与 gate：`python/benchmarks/results/reference/`
- 最终默认策略回归：`answers-multihop-release-smoke-v1.jsonl` 与 `answers-multihop-release-smoke-v1-report.json`
- 固定数据清单与许可证：`python/benchmarks/manifest.json`、`python/benchmarks/manifest.lock.json`
- 运行命令：`akh-benchmark run-multihop-v5 --confirm-live`。该命令会先做预检和 10 题 smoke，100 题门控失败时不会自动继续付费跑剩余 200 题。

原始公开数据、私人 Base URL、API Key、运行日志和 SQLite 状态均不进入仓库。
