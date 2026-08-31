# AgentKnowledgeHub Python 架构

## 系统边界

本仓库公开版本只包含 Python MVP。系统负责企业文档接入、知识抽取、混合检索问答和文件级 CDC 更新；Web UI、认证、多租户、Debezium、Java 和 Go 不属于当前交付范围。

```text
┌──────────────────────────────────────────────────────────────────────┐
│ FastAPI / Swagger                                                    │
│ upload · batch · ask · update · documents · health · stats           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│ Supervisor Agent（LangGraph）                                        │
│ validate ──route──> ingest graph | qa graph | update graph           │
└───────────────┬──────────────────┬──────────────────┬────────────────┘
                │                  │                  │
      ┌─────────▼────────┐ ┌───────▼────────┐ ┌──────▼──────────┐
      │ 文档解析 Agent    │ │ 智能问答 Agent  │ │ 知识更新 Agent   │
      │ 解析/OCR/分块     │ │ 规划/检索/回答   │ │ 幂等/差异/修复   │
      └─────────┬────────┘ └───────┬────────┘ └──────┬──────────┘
                │                  │                  │
                └──────────┬───────┴─────────┬────────┘
                           │ 知识抽取 Agent   │
                           │ 实体/关系/来源   │
                           └────────┬─────────┘
                                    │
        ┌───────────────────────────▼────────────────────────────┐
        │ SQLite 目录/Checkpoint · Chroma 三模态向量 · Neo4j 图谱 │
        │ Kafka CDC 事件 · SHA-256 受管图片资产                  │
        └────────────────────────────────────────────────────────┘
```

四个 Agent 是职责边界，不表示每个步骤都调用大模型：解析、分块、哈希差异、幂等、日期比较和一致性校验优先使用确定性代码；LLM 主要用于视觉描述、结构化知识抽取、查询规划和最终回答。

## 流水线一：文档入库

```text
file + MIME + doc_id
        │
        ▼
validate ─失败──────────────────────────────┐
        │                                    │
        ▼                                    │
parse → 标准化 Chunk + 受管图片资产          │
        │                                    │
        ▼                                    │
diff ──未变化──> commit                      │
        │ changed                            │
        ▼                                    │
embed → text/table/image 三集合              │
        │                                    │
        ▼                                    │
extract → 带 source_chunk_id 的实体/关系      │
        │                                    │
        ▼                                    │
graph_upsert → delete_removed → commit → verify
        │                                    │
        └────────任一阶段失败────────────> rollback
```

- 支持 PDF、DOCX、XLSX、CSV、PNG/JPG、TXT、Markdown、PPTX、HTML、JSON/JSONL、XML，共 11 类格式族。
- Chunk 约 500 token、80 token overlap，并保留页码、工作表、章节、坐标、JSONPath/XPath 和 modality。
- `chunk_id` 由文档、来源单元、规范化内容及重复序号稳定生成；重传时未变化块保持原 ID。
- 图片和扫描页始终走 OCR；视觉模型失败只记录降级，不使整份文档失败。

## 流水线二：智能问答

```text
question + mode + top_k + document_ids
                    │
                    ▼
             validate → QueryPlan
                    │
           ┌────────┴────────┐
           ▼                 ▼
  multimodal vector     Neo4j 1–3 hop
  text/table/image       参数化子图检索
           │                 │
           └────────┬────────┘
                    ▼
            strict weighted RRF
                    ▼
       document rerank（最多 10 篇）
                    ▼
        选中文档内复用查询向量二次检索
                    ▼
        chunk rerank + 槽位/文档多样性选证据
                    ▼
             evidence route
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
  时间比较工具    受限计算器    普通回答/拒答
       └────────────┴─────────────┘
                    ▼
      citation 必须映射真实 chunk → finalize
```

Chroma 使用 `knowledge_text_v2`、`knowledge_table_v2`、`knowledge_image_v2` 三个集合。文本与表格查询使用 BGE 文本向量，图片集合使用 VL 查询向量；集合内排名先归一化，再按 text `1.0`、table `0.95`、image `0.90` 做 RRF。向量结果与 Neo4j 结果继续用严格加权 RRF 融合，绝不执行 LLM 生成的任意 Cypher。

最终默认策略保留语义时间比较工具；实验性的通用 `ComparisonTool` 默认关闭，因为固定 100 题消融中它提高了证据召回，却降低了回答 EM。该工具可通过 `COMPARISON_TOOL_ENABLED=true` 单独复现实验。

## 流水线三：CDC 增量更新

```text
Watchdog（2 s debounce）/ 管理 API
                    │
                    ▼
Kafka event: event_id, operation, doc_id, hash, timestamp
                    │
                    ▼
validate_event → idempotency
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
      delete                  upsert
        └───────────┬───────────┘
                    ▼
 diff: added / changed / deleted / unchanged
                    ▼
 apply（仅处理变化块）→ verify
                    │
            ┌───────┴────────┐
            ▼                ▼
          commit           repair → verify
                              │ 仍失败
                              ▼
                             fail
```

SQLite 中事件状态按 `PENDING → APPLYING → COMMITTED/FAILED` 推进，同一 `event_id` 重放直接返回已提交结果。SQLite、Chroma、Neo4j 没有分布式事务，因此使用稳定 ID、幂等写入、提交后集合一致性校验和显式修复命令实现最终一致性。

## 安全与可观测性

- 上传限制默认 50 MB，并校验文件名、扩展名、MIME、文件签名和 OOXML 压缩包路径/体积。
- HTML 不抓取远程资源；XML 禁止 DTD/XXE；JSON/XML 有深度与节点数量上限。
- `/api/health/live` 只表示进程存活；`/api/health/ready` 检查 SQLite、Chroma、Neo4j、Kafka 和模型配置。
- Trace 只记录阶段、路由、候选数、重试和降级，不输出模型内部思维链、正文或密钥。
- `.env`、原始数据、运行日志、SQLite 文件和上传内容均不进入 Git。
