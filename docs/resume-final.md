# 简历项目经历（可直接使用）

项目名称：企业级多 Agent 知识管理系统（AgentKnowledgeHub）

时间：2026.01–2026.03

角色：核心开发（独立架构设计 + 后端与 AI 全链路开发）

技术栈：Python / LangGraph / LangChain / Neo4j / ChromaDB / FastAPI / Docker / Kafka

项目描述：

设计并开发 4-Agent 混合编排的企业文档知识管理系统，支持多模态文档解析、知识图谱构建、GraphRAG 带引用问答和 CDC 增量更新，实现文档接入、知识加工、检索问答与变更同步的完整生命周期管理。

核心工作：

- 设计文档解析、知识抽取、智能问答、知识更新 4 类职责 Agent，基于 LangGraph 实现文档入库、智能问答、增量更新三条多节点流水线，支持类型化状态、条件路由、并行检索、自动重试和 SQLite Checkpoint 恢复。
- 实现多模态 RAG 管道，统一处理 PDF、DOCX、XLSX、CSV、PPTX、图片、HTML、JSON、XML 等 11 类格式族；结合 OCR 与 Qwen3-VL 处理图片和扫描页，将文本、结构化表格、原始图片分别向量化并执行加权融合检索。
- 基于 Neo4j 构建带 Chunk 来源的知识图谱，融合 Chroma 多模态向量召回、参数化 1–3 跳子图检索、严格加权 RRF、BGE 两级精排和文档内二次检索；固定 100 题配对实验中，Hybrid 将 All-evidence Hit@10 从 45.33% 提升至 50.67%。
- 设计 Watchdog/API → Kafka → Consumer 的 CDC 链路，利用稳定 Chunk ID 计算 added/changed/deleted/unchanged，只处理变化内容；通过事件状态机、幂等消费和一致性修复协调 SQLite、ChromaDB、Neo4j。
- 建立可复现评测与工程质量体系，支持固定分层采样、查询计划缓存、断点续跑及 bootstrap 95% CI；126 个自动测试覆盖工作流路由、多格式解析、上传安全、Rerank 降级、多模态检索和 CDC 一致性，核心模块行覆盖率 87.94%。

## 面试时主动说明

- `50.67%` 是 All-evidence Hit@10，不是最终回答准确率；对应 Vector 基线是 `45.33%`。
- 同一 100 题实验中 Hybrid Answer EM 为 `57.33%`，Vector 为 `60.00%`，所以不能宣称图检索显著提升回答准确率。
- 项目证明的是完整工程链路和诚实的消融分析，不宣称未经压测的千级文档、2 秒 p95 或 95% CDC 加速。
