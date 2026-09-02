# 检索升级 v6：Parent/Child + BM25 + 邻域上下文

## 变更范围

- 文本检索子块从 500/80 调整为 350/50 token。
- 每个子块保存 900/100 token 的父块映射；精排时使用父块与前后一个子块，citation 仍引用原始子块。
- SQLite Catalog 增加 FTS5/BM25 索引，Dense 与 BM25 使用 Weighted RRF 融合后再进入 GraphRAG。
- Embedding 输入增加文件名、章节、页码、工作表等确定性上下文前缀，但 Chroma 保存的引用正文不变。
- 知识抽取增加 503/超时等传输错误的显式退避重试。

## 固定 10 题 Smoke 结果

数据为 MultiHop-RAG prepared retrieval 集的固定前 10 题，三组结果的 sample_id 完全一致。Top-K 均为 10，
Precision 按最终 10 个 Chunk 聚合后的唯一文档集合计算。

| 策略 | 文档 Precision@10 | Evidence Recall@10 | 文档 F1@10 | All-evidence Hit@10 | 检索 p50 | 检索 p95 |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Hybrid Weighted-RRF v2 | 27.75% | 94.17% | 42.44% | 80.00% | 1.215 s | 1.739 s |
| 新 Vector（Dense + BM25 + 两级精排） | 50.50% | 94.17% | 64.85% | 80.00% | 5.457 s | 9.277 s |
| 新 Hybrid（Vector + Graph） | 50.33% | 97.50% | 65.71% | 90.00% | 4.883 s | 9.084 s |

新 Hybrid 相比同版本 Vector 的 Recall 提高 3.33 个百分点，All-evidence Hit 提高 10 个百分点；10 题中有
1 题由缺一篇证据变为找齐。Precision 基本持平，表明这 10 题上的图候选没有明显放大文档噪声。

## 如何解释

这只是功能与方向性预检，样本数 10，不能据此更新简历中的正式 100 题数字，也不能做显著性结论。旧 v2
与新 v6 同题，但同时改变了分块、词法召回、上下文扩展和证据打包，因此“新旧提升”不能单独归因于 BM25。
Precision 大幅上升还与“每篇文档最多两个 Chunk”的最终打包有关：新结果通常覆盖 5–6 篇文档，而旧结果
接近 10 篇；有价值之处是文档数减少后 Recall 没有下降。

代价也很明确：两次外部 Rerank 加上更深候选池，使检索 p50/p95 高于旧 v2。本轮入库在抽取并发 8 时
遇到一次反代 503，降到并发 4 后完成。因此当前结论是“质量方向值得保留，但正式扩大到 100 题前需要做
分层消融和延迟优化”，而不是宣布升级已经全面优于旧版。

原始记录位于
`python/benchmarks/results/smoke/retrieval-parent-bm25-v6-10.jsonl`，历史对照位于
`python/benchmarks/results/smoke/retrieval-hybrid-weighted-rrf-v2-10.jsonl`。

## 下一轮建议

1. 固定同一索引和 QueryPlan，比较 Dense、Dense+BM25、+Parent/Neighbor、+Graph 四级消融。
2. 将文档级与 Chunk 级 Rerank 合批或按题型跳过一次 Rerank，目标先把检索 p95 降到 6 秒以内。
3. 通过 30 题开发集后再跑独立固定 100 题，并报告配对 bootstrap 95% CI。
4. Answer EM 与 Citation Recall 必须另跑回答实验，不能用本页检索指标代替回答准确率。
