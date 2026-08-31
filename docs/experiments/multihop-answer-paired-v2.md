# MultiHop-RAG 配对问答实验（Weighted-RRF v2）

## 实验目的

检验 Weighted-RRF Hybrid 检索已经取得的召回提升，能否继续转化为最终问答质量提升。实验不把 `Recall@10` 当作问答准确率，而是分别报告答案、拒答、引用和延迟指标。

## 数据与固定变量

- 语料：从 MultiHop-RAG 的 609 篇公开语料中按固定规则选择 100 篇。
- 问题：固定随机种子 42 抽取 300 题，其中 225 题有答案、75 题不可回答。
- 对照组：Vector-only。
- 实验组：Hybrid，即多模态向量候选与 Neo4j 图候选经过严格加权 RRF 融合。
- 两组都使用相同的 100 篇语料、500-token 分块、80-token overlap、TopK=10、生成模型、Embedding 模型和 `temperature=0`。
- 每道题只生成一次 QueryPlan，并缓存后同时交给 Vector 和 Hybrid。这样意图识别、查询改写和实体抽取完全相同，组间主要变量是检索方式。
- 运行顺序固定为同一道题先 Vector、后 Hybrid。模型服务仍可能存在微小非确定性，因此结论以 300 题配对结果和 bootstrap 置信区间为准。

## 评分方法

有答案的 225 题报告：

- `Exact Match`：规范化后与标准答案完全相同，最严格，作为主要准确率指标之一。
- `Token F1`：预测与标准答案的 token 重叠 F1，允许部分命中，作为主要连续指标。
- `Official Overlap Accuracy`：复现 MultiHop-RAG 仓库的宽松规则，只要预测与标准答案存在一个空格分词交集就记为正确。该指标仅用于兼容对比，不能单独作为简历中的“准确率”。
- `Citation Precision / Recall / F1`：按文档 ID 比较模型实际引用与数据集标注的证据文档。

全部 300 题报告：

- `Rejection F1`：把“不可回答”视作正类，检查 75 道无答案题能否拒答，同时惩罚对可回答问题的错误拒答。
- `Answerability Accuracy`：回答/拒答二分类是否与标签一致。
- 检索、生成及端到端延迟的 p50/p95。
- 模型调用次数、输入/输出 token 和错误数。

Vector 与 Hybrid 的 EM、Token F1、宽松 Accuracy 和 Citation Recall 使用逐题配对 bootstrap（2,000 次、seed=42）计算 95% 置信区间。只有提升值为正且区间下界大于 0，才表述为有可信提升。

## 预注册判断标准

- 核心判断：Hybrid Token F1 高于 Vector，且配对 95% CI 下界大于 0。
- 辅助判断：Hybrid EM 不低于 Vector；Citation Recall 不因答案生成环节明显丢失。
- 可用性门槛：Hybrid Rejection F1 不低于 0.75，Citation Recall 不低于 0.85。
- 如果 Hybrid Recall@10 明显提高但答案 F1 没提高，将重点检查无关上下文、生成提示、引用筛选和 `TopK=10` 带来的噪声，而不会继续把召回提升等同于问答提升。
- 未达到门槛的结果照常保留并分析，不调整样本或删除失败题。

## 调用量预估

- 最少 300 次查询规划调用，加上 300 × 2 次答案生成，共 900 次 Chat API 调用；重试会增加调用数。
- 每种检索都要生成文本查询向量和 VL 文本查询向量。根据每题 QueryPlan 含 1–3 条查询，预计 1,200–3,600 次 Embedding 请求。
- 旧运行记录中每次答案生成平均约 5,325 个输入 token、154 个输出 token。据此估计 600 次答案生成约消耗 319.5 万输入 token 和 9.2 万输出 token；加上 300 次查询规划后，总 Chat 输入约为 330 万 token 量级。
- 本地未配置反代生成模型的单价，因此只记录真实 token，不虚构人民币费用；Embedding 成本按运行时配置价格汇总。

## 产物与断点续跑

- 原始逐题结果：`python/benchmarks/results/reference/answers-multihop-paired-weighted-rrf-v2.jsonl`
- 每题共享 QueryPlan：`python/benchmarks/results/reference/answers-multihop-paired-weighted-rrf-v2-query-plans.jsonl`
- 指标与配置快照：`python/benchmarks/results/reference/answers-multihop-paired-weighted-rrf-v2-summary.json`
- 运行日志：`python/benchmarks/results/multihop-answer-paired-weighted-rrf-v2-*/stdout.log` 与 `stderr.log`

每条结果以 `sample_id:mode` 作为幂等键，每个 QueryPlan 以 `sample_id` 作为幂等键。任务中断后使用相同输出路径重启，只补跑缺失记录，不会从头重复计费。

## 结论有效性边界

- 当前只使用 100 篇语料，不代表完整 609 篇语料或生产环境规模。
- MultiHop-RAG 的证据标签是文档级，不验证引用是否精确到正确句子；句子级引用质量后续需要人工抽检。
- 官方 overlap 指标非常宽松，最终结论以 EM、Token F1、拒答 F1、Citation 指标及其置信区间为主。
- 本实验验证的是当前固定配置，不能直接外推为所有模型、所有 TopK 或所有企业文档都能获得相同提升。
