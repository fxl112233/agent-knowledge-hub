# AgentKnowledgeHub（Python）

面向企业文档的 4-Agent 知识管理 MVP。系统用 LangGraph 编排文档解析、知识抽取、
GraphRAG 问答和 CDC 更新 Agent，以 FastAPI 提供接口，ChromaDB 保存多模态向量，
Neo4j 保存带来源的知识图谱，Kafka 传递增量事件，SQLite 保存目录、版本和检查点。

> 本次实现和验收只覆盖 `python/`。Java、Go、Web UI、认证、PGVector、Debezium 不在范围内。

快速入口：[架构说明](docs/architecture.md) · [量化评测报告](docs/benchmark-report.md) · [5 分钟演示](docs/demo-guide.md) · [面试讲解](docs/interview-guide.md) · [简历写法](docs/resume-final.md)

## 已实现能力

### 4-Agent 与三条真实 LangGraph 流水线

#### 整体架构

```mermaid
flowchart TB
    USER["调用方 / Swagger"]
    WATCHER["Watchdog<br/>监听文件创建、修改、删除"]

    subgraph ACCESS["接入与事件层"]
        UPLOAD["POST /api/ingest/upload<br/>文件字节 + 文件名 + MIME"]
        ASK["POST /api/qa/ask<br/>question + top_k + mode + document_ids"]
        ADMIN["POST /api/admin/update<br/>doc_id + change_type"]
        PRODUCER["CDC Producer<br/>构造统一 CDCEvent"]
        KAFKA["Kafka: document change topic<br/>持久化增量事件"]
        CONSUMER["Kafka Consumer<br/>手动提交 offset"]
    end

    subgraph ORCHESTRATION["LangGraph 编排层"]
        SUPERVISOR["Supervisor Graph<br/>workflow + payload + thread_id"]
        INGEST_GRAPH["Ingest 子图<br/>IngestState"]
        QA_GRAPH["QA 子图<br/>QAState"]
        UPDATE_GRAPH["Update 子图<br/>UpdateState"]
        CHECKPOINT["SQLite Checkpointer<br/>thread_id + 节点状态"]
    end

    subgraph AGENTS["4 个业务 Agent"]
        PARSER["Document Parser Agent<br/>11 类格式、OCR、视觉降级"]
        EXTRACTOR["Knowledge Extract Agent<br/>实体、关系、来源校验"]
        QA_AGENT["QA Agent<br/>规划、检索、回答、引用"]
        UPDATE_AGENT["Knowledge Update Agent<br/>幂等、重试、增量应用"]
    end

    subgraph MODELS["模型服务"]
        LLM["生成模型 API<br/>计划、抽取、回答"]
        TEXT_EMBED["BAAI/bge-m3<br/>文本 / 表格向量"]
        VISION["Qwen3-VL-Instruct<br/>图片客观描述"]
        VL_EMBED["Qwen3-VL-Embedding<br/>原始图片 / 文本查询向量"]
    end

    subgraph DATA["数据与基础设施层"]
        CATALOG["SQLite Catalog<br/>Document / Chunk / Event"]
        CHROMA["ChromaDB<br/>text_v2 / table_v2 / image_v2 / legacy"]
        NEO4J["Neo4j<br/>Entity / Chunk / RELATION / MENTIONED_IN"]
        ASSETS["受管资产目录<br/>SHA-256 图片文件"]
    end

    USER --> UPLOAD
    USER --> ASK
    USER --> ADMIN
    WATCHER -->|"file_path + operation + observed_hash"| PRODUCER
    ADMIN -->|"先写 Event=PENDING"| CATALOG
    ADMIN --> PRODUCER
    PRODUCER -->|"CDCEvent JSON"| KAFKA
    KAFKA -->|"反序列化 CDCEvent"| CONSUMER

    UPLOAD -->|"file_paths + doc_ids + mime_types"| INGEST_GRAPH
    ASK -->|"question + top_k + mode + document_ids"| QA_GRAPH
    CONSUMER -->|"events"| UPDATE_GRAPH
    SUPERVISOR -. "统一编程入口：按 workflow 确定性路由" .-> INGEST_GRAPH
    SUPERVISOR -. "统一编程入口：按 workflow 确定性路由" .-> QA_GRAPH
    SUPERVISOR -. "统一编程入口：按 workflow 确定性路由" .-> UPDATE_GRAPH

    INGEST_GRAPH --> PARSER
    INGEST_GRAPH --> EXTRACTOR
    QA_GRAPH --> QA_AGENT
    UPDATE_GRAPH --> UPDATE_AGENT
    UPDATE_AGENT -->|"INSERT / UPDATE 复用入库服务"| PARSER
    UPDATE_AGENT -->|"INSERT / UPDATE 复用入库服务"| EXTRACTOR

    PARSER --> VISION
    EXTRACTOR --> LLM
    QA_AGENT --> LLM
    INGEST_GRAPH --> TEXT_EMBED
    INGEST_GRAPH --> VL_EMBED
    QA_AGENT --> TEXT_EMBED
    QA_AGENT --> VL_EMBED

    PARSER -->|"DocumentChunk + metadata"| ASSETS
    INGEST_GRAPH -->|"文档、版本、chunk 目录"| CATALOG
    INGEST_GRAPH -->|"多模态向量"| CHROMA
    INGEST_GRAPH -->|"实体、关系、来源 chunk"| NEO4J
    QA_AGENT -->|"相似度候选"| CHROMA
    QA_AGENT -->|"1 至 3 跳子图候选"| NEO4J
    UPDATE_AGENT -->|"事件状态与文档版本"| CATALOG
    UPDATE_AGENT -->|"增量新增 / 删除"| CHROMA
    UPDATE_AGENT -->|"增量新增 / 删除"| NEO4J

    INGEST_GRAPH <--> CHECKPOINT
    QA_GRAPH <--> CHECKPOINT
    UPDATE_GRAPH <--> CHECKPOINT
```

图中实线是当前 HTTP/Kafka 的实际调用路径。FastAPI 的三个接口分别直接调用公开子图；
`Supervisor Graph` 是统一的可编程入口，根据显式 `workflow=ingest|qa|update` 做确定性分发，
不让 LLM 猜测应该进入哪条业务流水线。

#### 流水线 1：文档入库

```mermaid
flowchart TD
    REQUEST["上传请求<br/>文件字节 + file_name + MIME"]
    API_SAFE["API 上传安全校验<br/>大小、路径、扩展名、MIME、签名、压缩包"]
    STATE["IngestState<br/>file_paths + doc_ids + mime_types"]
    VALIDATE["validate<br/>计算 SHA-256、读取旧 Document/Chunk、申请版本"]
    TX["IngestTransaction<br/>doc_id + version + content_hash + old_chunks"]
    PARSE["parse<br/>11 类解析 + OCR + 可选视觉描述"]
    PARSED["DocumentChunk[]<br/>content + chunk_id + page/sheet/slide/坐标 + modality + asset_path"]
    DIFF["diff<br/>按稳定 chunk_id 比较新旧版本"]
    SETS["added_ids / removed_ids / unchanged_ids<br/>added_chunks"]
    CHANGED{"整份文档内容是否未变化？"}

    subgraph VECTOR_WRITE["只处理 added_chunks"]
        MODALITY{"chunk.modality"}
        TEXT_TABLE["text / table<br/>构造文本或带坐标表格文本"]
        IMAGE["image<br/>读取 SHA-256 受管原图"]
        BGE["BAAI/bge-m3"]
        VL["Qwen3-VL-Embedding"]
        TEXT_DB["Chroma text_v2"]
        TABLE_DB["Chroma table_v2"]
        IMAGE_DB["Chroma image_v2"]
    end

    EXTRACT["extract<br/>新增块送入结构化知识抽取"]
    FACTS["ExtractionResult[]<br/>Entity + RELATION + confidence + source_chunk_id"]
    GRAPH_UPSERT["graph_upsert<br/>幂等写入 Chunk、Entity、MENTIONED_IN、RELATION"]
    NEO4J["Neo4j"]
    DELETE_REMOVED["delete_removed<br/>仅删除 removed_ids"]
    CHROMA["Chroma 三个 v2 集合及兼容旧集合"]
    COMMIT["commit<br/>提交 Document/Chunk 目录并裁剪失效资产"]
    SQLITE["SQLite Catalog<br/>Document=READY + 当前版本 ChunkRecord"]
    VERIFY["verify<br/>比较 Catalog、Chroma、Neo4j 的 chunk_id 集合"]
    OK["IngestResponse<br/>版本、增删/未变块数、实体/关系数、模态数、耗时"]
    RETRY["节点内自动重试<br/>普通阶段最多 2 次"]
    ROLLBACK["rollback<br/>补偿清理本次新增写入、保留旧资产、Document=FAILED"]
    ERROR["4xx 解析错误或 502 入库失败<br/>错误文本不进入知识库"]

    REQUEST --> API_SAFE
    API_SAFE -->|"保存到 uploads/doc_id/file_name"| STATE
    STATE --> VALIDATE
    VALIDATE --> TX
    TX --> PARSE
    PARSE --> PARSED
    PARSED --> DIFF
    DIFF --> SETS
    SETS --> CHANGED

    CHANGED -->|"是：已有 READY 且 hash 相同"| COMMIT
    CHANGED -->|"否"| MODALITY
    MODALITY -->|"text"| TEXT_TABLE
    MODALITY -->|"table"| TEXT_TABLE
    MODALITY -->|"image"| IMAGE
    TEXT_TABLE --> BGE
    BGE -->|"文本块向量"| TEXT_DB
    BGE -->|"表格块向量"| TABLE_DB
    IMAGE --> VL
    VL -->|"原始图片向量"| IMAGE_DB
    TEXT_DB --> EXTRACT
    TABLE_DB --> EXTRACT
    IMAGE_DB --> EXTRACT
    EXTRACT --> FACTS
    FACTS --> GRAPH_UPSERT
    GRAPH_UPSERT --> NEO4J
    NEO4J --> DELETE_REMOVED
    DELETE_REMOVED -->|"removed chunk_id"| CHROMA
    DELETE_REMOVED -->|"删除来源关系和无来源孤立实体"| NEO4J
    DELETE_REMOVED --> COMMIT
    COMMIT --> SQLITE
    SQLITE --> VERIFY
    CHROMA --> VERIFY
    NEO4J --> VERIFY
    VERIFY -->|"三方 ID 完全一致"| OK

    VALIDATE -. "校验失败" .-> ERROR
    PARSE -. "异常" .-> RETRY
    DIFF -. "异常" .-> RETRY
    MODALITY -. "Embedding 异常" .-> RETRY
    EXTRACT -. "抽取异常" .-> RETRY
    GRAPH_UPSERT -. "写图异常" .-> RETRY
    DELETE_REMOVED -. "删除异常" .-> RETRY
    RETRY -. "耗尽" .-> ROLLBACK
    COMMIT -. "提交异常" .-> ROLLBACK
    VERIFY -. "不一致" .-> ROLLBACK
    ROLLBACK --> ERROR
```

这里的关键是 `diff` 之后只有 `added_chunks` 会重新做 Embedding 和知识抽取；
`unchanged_ids` 不触碰，`removed_ids` 才从向量库和图谱中删除。

#### 流水线 2：智能问答

```mermaid
flowchart TD
    REQUEST["QuestionRequest<br/>question + top_k + mode + document_ids"]
    VALIDATE["validate<br/>问题非空、top_k 1 至 20、mode 合法"]
    PLAN["plan<br/>一次 LLM 结构化规划"]
    QUERY_PLAN["QueryPlan<br/>intent + answer_type + evidence_slots 最多 3 个<br/>target_relation + entities + needs_calculation"]

    VMODE{"mode 允许向量检索？<br/>vector / hybrid"}
    GMODE{"mode 允许图谱检索？<br/>graph / hybrid"}
    VSKIP["空 Vector RetrievalResult<br/>trace=vector:skipped"]
    GSKIP["空 Graph RetrievalResult<br/>trace=graph:skipped"]

    subgraph VECTOR_PATH["向量分支：各改写查询并行执行"]
        TEXT_QUERY["普通文本查询向量<br/>BAAI/bge-m3"]
        VL_QUERY["VL 文本查询向量<br/>Qwen3-VL-Embedding"]
        TEXT_SEARCH["检索 text_v2"]
        TABLE_SEARCH["检索 table_v2"]
        IMAGE_SEARCH["检索 image_v2<br/>实现文本查原图"]
        LEGACY_SEARCH["兼容检索 legacy<br/>过滤与 v2 重复的 chunk_id"]
        MODAL_FUSE["模态内加权 RRF<br/>text 1.0 / table 0.95 / image 0.90"]
        VRESULT["Vector RetrievalResult<br/>contexts + modality candidate counts"]
    end

    subgraph GRAPH_PATH["图谱分支：只执行参数化查询，不执行 LLM 生成 Cypher"]
        HOPS["intent 映射跳数<br/>factoid/procedural=1<br/>analytical/comparative=2<br/>exploratory=3"]
        GRAPH_QUERY["entities + hops + document_ids + limit"]
        GRAPH_DB["Neo4j 子图遍历<br/>Entity → RELATION → Entity → MENTIONED_IN → Chunk"]
        GRESULT["Graph RetrievalResult<br/>来源 chunk + 路径实体/谓词 + score"]
    end

    RRF_FUSE["rrf_fuse<br/>Vector/Graph 严格 Weighted RRF<br/>vector=1.0 / graph=0.85，不混合原始分数"]
    RERANK_DOCS["rerank_docs<br/>证据槽位并行 BGE 精排 40 个候选，最多选择 10 篇文档"]
    DOC_REFINE["document_refine<br/>复用已有查询向量，只在选中文档内二次检索<br/>每查询每文档最多 6 个候选，总计最多 80 chunk"]
    RERANK_CHUNKS["rerank_chunks<br/>段落级 BGE 精排 + 证据槽位优先打包<br/>每文档最多 2 个 chunk，最终截取 top_k"]
    ROUTE{"回答路由"}
    ABSTAIN["abstain<br/>无 contexts，生成 answerable=false"]
    CALCULATOR["calculator 路由<br/>答案阶段仅允许受限算术表达式"]
    TEMPORAL["temporal_tool<br/>提取左右事实/日期/引用并确定性比较<br/>统一输出 Yes/No；缺证据则拒答"]
    ANSWER["answer<br/>LLM 只依据检索 contexts 生成结构化答案"]
    DRAFT["QAResult<br/>answer + answerable + evidence_score + citation chunk_ids + usage"]
    CITATION["citation_validate<br/>引用必须属于本次真实检索 chunk；可回答却无引用则降为不可回答"]
    FINALIZE["finalize<br/>合并 retrieval/generation/total 耗时与高层 trace"]
    RESPONSE["QuestionResponse<br/>答案 + citations + modality + score + token 用量"]
    FAIL["fail<br/>重试耗尽后返回问答失败"]

    REQUEST --> VALIDATE
    VALIDATE --> PLAN
    PLAN --> QUERY_PLAN
    QUERY_PLAN --> VMODE
    QUERY_PLAN --> GMODE

    VMODE -->|"是"| TEXT_QUERY
    VMODE -->|"是"| VL_QUERY
    VMODE -->|"否：mode=graph"| VSKIP
    TEXT_QUERY --> TEXT_SEARCH
    TEXT_QUERY --> TABLE_SEARCH
    TEXT_QUERY --> LEGACY_SEARCH
    VL_QUERY --> IMAGE_SEARCH
    TEXT_SEARCH --> MODAL_FUSE
    TABLE_SEARCH --> MODAL_FUSE
    IMAGE_SEARCH --> MODAL_FUSE
    LEGACY_SEARCH --> MODAL_FUSE
    MODAL_FUSE --> VRESULT

    GMODE -->|"是"| HOPS
    GMODE -->|"否：mode=vector"| GSKIP
    HOPS --> GRAPH_QUERY
    GRAPH_QUERY --> GRAPH_DB
    GRAPH_DB --> GRESULT

    VRESULT --> RRF_FUSE
    VSKIP --> RRF_FUSE
    GRESULT --> RRF_FUSE
    GSKIP --> RRF_FUSE
    RRF_FUSE --> RERANK_DOCS
    RERANK_DOCS --> DOC_REFINE
    DOC_REFINE --> RERANK_CHUNKS
    RERANK_CHUNKS --> ROUTE
    ROUTE -->|"contexts 为空"| ABSTAIN
    ROUTE -->|"target_relation 为时间/变化关系"| TEMPORAL
    ROUTE -->|"有证据且 needs_calculation=true"| CALCULATOR
    ROUTE -->|"有证据且无需计算"| ANSWER
    CALCULATOR --> ANSWER
    ABSTAIN --> DRAFT
    TEMPORAL --> DRAFT
    ANSWER --> DRAFT
    DRAFT --> CITATION
    CITATION --> FINALIZE
    FINALIZE --> RESPONSE

    VALIDATE -. "非法请求" .-> FAIL
    PLAN -. "最多 2 次仍失败" .-> FAIL
    VRESULT -. "向量分支错误；hybrid 可退化到图谱" .-> RRF_FUSE
    GRESULT -. "图谱分支错误；hybrid 可退化到向量" .-> RRF_FUSE
    RERANK_DOCS -. "超时/429/5xx 重试后回退 RRF" .-> DOC_REFINE
    RERANK_CHUNKS -. "重排失败回退候选顺序" .-> ROUTE
    ANSWER -. "最多 2 次仍失败" .-> FAIL
```

这里有三层不同的“路由”：Supervisor 选择业务子图，API 的 `mode` 决定向量/图谱分支是否工作，
`QueryPlan.intent` 决定图查询跳数，并与 `target_relation` 一起选择时间工具；最后再根据证据是否
充分、是否需要计算选择时间比较、计算器、普通回答或拒答。
混合模式没有硬编码“图谱更准确所以权重更高”。

#### 流水线 3：CDC 增量更新

```mermaid
flowchart TD
    WATCH["Watchdog<br/>2 秒去抖后的 created / modified / deleted"]
    ADMIN["管理 API<br/>doc_id + modified / deleted"]
    EVENT["CDCEvent<br/>event_id + operation + doc_id + file_path + observed_hash + timestamp"]
    PENDING["SQLite EventRecord=PENDING<br/>管理 API 在发布前写入"]
    PRODUCER["Kafka Producer<br/>key=event_id，value=CDCEvent JSON"]
    KAFKA["Kafka topic"]
    CONSUMER["Kafka Consumer<br/>反序列化事件，不自动提交 offset"]
    STATE["UpdateState<br/>events[]"]
    VALIDATE["validate_event<br/>只接受 INSERT / UPDATE / DELETE"]
    IDEMPOTENCY["idempotency<br/>按 event_id 查询 SQLite EventRecord"]
    DUPLICATE{"是否已 COMMITTED？"}
    DUP_RESULT["duplicate_result=true<br/>不再写 Chroma / Neo4j / Catalog"]
    OPERATION{"活动事件类型"}
    DELETE_ROUTE["delete 路由标记"]
    UPSERT_ROUTE["upsert 路由标记"]
    DIFF_TRACE["diff 节点<br/>记录本批 upsert 数；真正 chunk diff 在 apply 内执行"]
    APPLY["apply<br/>逐事件调用 KnowledgeUpdateAgent.process_event"]
    CLAIM["原子 claim_event<br/>PENDING/FAILED → APPLYING，attempts + 1"]
    EVENT_OP{"event.operation"}

    subgraph DELETE_APPLY["DELETE 实际应用"]
        DELETE_DOC["IngestionService.delete(doc_id)"]
        DELETE_VECTOR["按 doc_id 删除所有 Chroma chunk"]
        DELETE_GRAPH["按 doc_id 删除图谱来源与孤立实体"]
        DELETE_CATALOG["删除 Catalog 文档、chunk 与受管资产"]
    end

    subgraph UPSERT_APPLY["INSERT / UPDATE 实际应用"]
        REINGEST["IngestionService.ingest(file_path, doc_id)"]
        PARSE_DIFF["重新解析并计算<br/>added / removed / unchanged"]
        ADD_ONLY["只对 added 做 Embedding、抽取和图谱 upsert"]
        REMOVE_ONLY["只对 removed 删除向量与图谱来源"]
        KEEP["unchanged 保留原 chunk_id、向量和图谱来源"]
        VERSION["成功后提交新文档版本和 ChunkRecord"]
    end

    FINISH{"单事件执行成功？"}
    COMMITTED["finish_event<br/>EventRecord=COMMITTED"]
    FAILED["finish_event<br/>EventRecord=FAILED + error"]
    RETRY{"attempts 小于 3？"}
    RESULT["UpdateResult[]<br/>增删向量数、未变块数、版本、实体/关系数"]
    VERIFY["verify<br/>active_results 全部成功，且 Catalog 期望 chunk_id = Chroma chunk_id = Neo4j chunk_id"]
    CONSISTENT{"三方一致？"}
    REPAIR["repair 一次<br/>DELETE 重新清理；UPDATE 按 Catalog 重建缺失存储"]
    REPAIR_DONE{"修复后是否一致？"}
    COMMIT["workflow commit<br/>合并 duplicate_results + active_results"]
    OFFSET["Consumer 同步提交 Kafka offset"]
    FAIL["workflow fail<br/>保留 FAILED 事件；offset 不提交，等待重放/修复"]

    WATCH --> EVENT
    ADMIN --> EVENT
    ADMIN --> PENDING
    PENDING --> PRODUCER
    EVENT --> PRODUCER
    PRODUCER --> KAFKA
    KAFKA --> CONSUMER
    CONSUMER --> STATE
    STATE --> VALIDATE
    VALIDATE --> IDEMPOTENCY
    VALIDATE -. "非法 operation 或空事件" .-> FAIL
    IDEMPOTENCY --> DUPLICATE
    DUPLICATE -->|"是"| DUP_RESULT
    DUPLICATE -->|"否：active_events"| OPERATION
    DUP_RESULT --> COMMIT

    OPERATION -->|"全是 DELETE"| DELETE_ROUTE
    OPERATION -->|"包含 INSERT / UPDATE"| UPSERT_ROUTE
    DELETE_ROUTE --> DIFF_TRACE
    UPSERT_ROUTE --> DIFF_TRACE
    DIFF_TRACE --> APPLY
    APPLY --> CLAIM
    CLAIM --> EVENT_OP
    EVENT_OP -->|"DELETE"| DELETE_DOC
    DELETE_DOC --> DELETE_VECTOR
    DELETE_DOC --> DELETE_GRAPH
    DELETE_DOC --> DELETE_CATALOG
    DELETE_VECTOR --> FINISH
    DELETE_GRAPH --> FINISH
    DELETE_CATALOG --> FINISH

    EVENT_OP -->|"INSERT / UPDATE"| REINGEST
    REINGEST --> PARSE_DIFF
    PARSE_DIFF --> ADD_ONLY
    PARSE_DIFF --> REMOVE_ONLY
    PARSE_DIFF --> KEEP
    ADD_ONLY --> VERSION
    REMOVE_ONLY --> VERSION
    KEEP --> VERSION
    VERSION --> FINISH

    FINISH -->|"是"| COMMITTED
    FINISH -->|"否"| FAILED
    FAILED --> RETRY
    RETRY -->|"是：指数退避后重领"| CLAIM
    RETRY -->|"否"| RESULT
    COMMITTED --> RESULT
    RESULT --> VERIFY
    VERIFY --> CONSISTENT
    CONSISTENT -->|"是"| COMMIT
    CONSISTENT -->|"否且尚未修复"| REPAIR
    REPAIR --> REPAIR_DONE
    REPAIR_DONE -->|"是"| COMMIT
    REPAIR_DONE -->|"否"| FAIL
    COMMIT --> OFFSET
```

CDC 中的 `delete / upsert / diff` 是工作流的显式路由与追踪节点，真正的单事件修改在 `apply`
里执行。这样 Kafka 重放同一个 `event_id` 时会先命中 `COMMITTED`，直接返回重复结果，不会重复写库。

- 普通节点最多尝试 2 次，CDC 最多尝试 3 次。
- SQLite Checkpointer 支持流程恢复；`trace` 只记录阶段、路由、重试和错误类型，不返回思维链。
- 问答中的多模态向量检索和 Neo4j 1–3 跳参数化查询并行执行，不执行模型生成的 Cypher。
- 数值问题走受限计算器；无充分证据时返回 `answerable=false`；引用必须映射到真实 chunk。

### 11 类文档与真多模态处理

| 格式族 | 扩展名 | 保留信息 |
|---|---|---|
| PDF | `.pdf` | 页码、文本、表格；低文本页 OCR/视觉 |
| Word | `.docx` | 标题、段落、表格、内嵌图片 |
| Excel | `.xlsx` | 工作表、行列、单元格坐标、表头 |
| CSV | `.csv` | 行列、表头、编码 |
| 图片 | `.png`、`.jpg`、`.jpeg` | OCR、视觉描述、原始图片向量 |
| 文本 | `.txt` | UTF-8/GB18030 等编码检测 |
| Markdown | `.md`、`.markdown` | 标题层级和正文 |
| PowerPoint | `.pptx` | 幻灯片、标题、文本框、表格、图片 |
| HTML | `.html`、`.htm` | 标题、正文、表格；不抓取远程资源 |
| JSON | `.json`、`.jsonl` | JSONPath、深度和节点数限制 |
| XML | `.xml` | XPath；安全解析并禁止 XXE |

图片、扫描页和 Office 内嵌图片以 SHA-256 命名保存为受管资产。OCR 始终执行；启用视觉时，
`Qwen/Qwen3-VL-8B-Instruct` 以 `detail=low` 补充客观描述，失败自动降级到 OCR，不使整份
文档失败。删除文档时会同步清理资产。

默认约 500 token、80 token overlap 分块，并保留页码、章节、工作表、坐标、模态、版本和
内容哈希。`chunk_id` 由文档、来源单元、规范化内容和重复序号确定，未变化块跨版本保持稳定。

### 独立向量化与 GraphRAG 融合

ChromaDB 使用三个 v2 集合：

- `knowledge_text_v2`：文本，使用 `BAAI/bge-m3`。
- `knowledge_table_v2`：带表头及坐标的结构化表格文本，使用 `BAAI/bge-m3`。
- `knowledge_image_v2`：原始图片，使用 `Qwen/Qwen3-VL-Embedding-8B`，不是只嵌入 OCR 文本。

查询同时生成普通文本向量和 VL 文本查询向量，并行检索三个集合；集合内排名归一化后按
文本 `1.0`、表格 `0.95`、图片 `0.90` 做加权 RRF，再与 Neo4j 子图结果融合。迁移期间仍可
读取旧 `knowledge_chunks` 集合，但同一 chunk 不会因同时存在于新旧集合而重复加分。
向量与图谱分支使用严格 Weighted RRF（默认权重 `1.0/0.85`），不混合不同来源的原始分数。
融合后的 40 个候选由硅基流动 `BAAI/bge-reranker-v2-m3` 按必需证据槽位并行精排；精排排名与
原始 Weighted-RRF 排名再次通过 RRF 融合，避免覆盖向量/图谱召回信号。系统复用首次生成的
文本/VL 查询向量，只在最多 10 篇候选文档内二次检索。最终 Top-K 先覆盖必需证据槽位，再按
融合分数自适应回填，每篇文档最多两个 chunk，不强制每篇候选文档都进入上下文。时间比较缺少
一侧证据时允许执行一次复用查询向量的深检索。Rerank 超时、429 或 5xx 最多重试两次，仍失败则
回退 RRF 候选，不中断回答。

实验性的通用 ComparisonTool 被保留用于消融，但默认 `COMPARISON_TOOL_ENABLED=false`。固定
100 题实验显示它提高了跨文档证据完整性，却因过度拒答降低 Answer EM；默认链路继续使用语义
明确的时间比较工具。这个取舍及失败案例见[量化评测报告](docs/benchmark-report.md)。

Neo4j 使用 `Entity`、`Chunk`、`MENTIONED_IN` 和通用 `RELATION` 来源模型；实体和关系都保留
`source_chunk_id`。删除 chunk 时同步删除向量、来源节点及关系，仅在没有其他来源时删除孤立实体。

### CDC 增量更新

Watchdog 和管理 API 只生产统一事件，Kafka Consumer 执行更新。事件包含 `event_id`、
`operation`、`doc_id`、`file_path`、`observed_hash`、`timestamp`，支持 2 秒去抖和幂等重放。

更新时按稳定 ID/hash 计算 `added / changed / deleted / unchanged`，只嵌入和抽取新增或变化块，
只删除消失块。跨 SQLite、ChromaDB、Neo4j 采用幂等写入和
`PENDING → APPLYING → COMMITTED/FAILED` 状态；一致性修复和全量重建命令用于恢复。

### 上传安全

默认文件上限 50 MB。API 校验文件名、目录穿越、扩展名、MIME、文件签名、压缩包成员路径、
成员数量、解压总大小及加密状态，并限制 PDF 页数、工作表数量、JSON/XML 深度和节点数。
HTML 不访问外部资源，XML 禁止实体展开；解析失败返回明确 4xx，不写入错误文本。

## 当前验证状态

以下是 2026-08-31 的工程验证与固定 MultiHop-RAG 实验结果：

| 项目 | 实测结果 |
|---|---:|
| 自动化测试 | 126 passed，1 个真实 API 用例按标记排除 |
| 核心模块行覆盖率 | 87.94% |
| Ruff | All checks passed |
| Docker Compose | FastAPI、ChromaDB、Neo4j、Kafka 4/4 healthy |
| Docker 11 格式端到端测试 | 1 passed（Fake 模型，不计费） |
| 最终默认链路固定回归 | 20/20 Vector/Hybrid 记录，契约检查全部通过，0 业务错误，0 Rerank fallback |
| 真实视觉预检 | 成功，描述 401 字符 |
| 真实 VL Embedding 预检 | 图片/文本均 1024 维，2 calls / 42 input tokens |
| 真实预检总耗时 | 6850.283 ms |

| 固定 100 题配对指标 | Vector | Hybrid |
|---|---:|---:|
| Evidence Recall@10 | 74.44% | 76.78% |
| All-evidence Hit@10 | 45.33% | 50.67% |
| Answer EM | 60.00% | 57.33% |
| Citation Recall | 40.67% | 43.78% |

Hybrid 的 All-evidence Hit@10 提升 `5.34 pp`，说明图候选提高了跨文档证据完整性；但 Answer EM
下降 `2.67 pp`，其配对 bootstrap 95% CI 为 `[-10.67, 5.33] pp`，不支持“GraphRAG 显著提升
回答准确率”的结论。完整设置、指标定义、版本消融与失败分析见
[docs/benchmark-report.md](docs/benchmark-report.md)。

CDC 性能和生产规模 API 压测尚未完成，因此本项目不声明“提升 22%”、“准确率 94%”、
“更新效率提升 95%”或“支持千级文档实时问答”等未经验证的数字。

## 配置与启动

要求 Python 3.12、Docker Desktop / Compose；OCR 使用 Tesseract `chi_sim+eng`，Docker 镜像已安装。

```powershell
Copy-Item python/.env.example python/.env
```

在 `python/.env` 中填写密钥，密钥不要提交：

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_API_KEY=
LLM_MODEL=

EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
SILICONFLOW_VL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B

VISION_ENABLED=true
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
MODALITY_TEXT_WEIGHT=1.0
MODALITY_TABLE_WEIGHT=0.95
MODALITY_IMAGE_WEIGHT=0.90
RRF_CONSTANT=60
HYBRID_VECTOR_WEIGHT=1.0
HYBRID_GRAPH_WEIGHT=0.85
HYBRID_MAX_CHUNKS_PER_DOCUMENT=1
RERANK_ENABLED=true
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_K=40
RERANK_TOP_DOCUMENTS=10
RERANK_LOCAL_CANDIDATES_PER_QUERY=6
RERANK_MAX_LOCAL_CANDIDATES=80
RERANK_MAX_CHUNKS_PER_DOCUMENT=2
RERANK_TIMEOUT_SECONDS=30
RERANK_MAX_RETRIES=2
COMPARISON_TOOL_ENABLED=false
ANSWER_MAX_CONTEXT_CHUNKS=8
ANSWER_MAX_CONTEXT_CHARS=18000
```

示例假设 OpenAI-compatible 服务在宿主机 8001 端口。Docker 中默认改写为
`http://host.docker.internal:8001/v1`；使用其他地址时，在执行 Compose 前设置
`DOCKER_LLM_BASE_URL`。`.env` 已从 Git 和 Docker build context 排除。

Dockerfile 默认使用阿里云 PyPI 镜像；也可临时覆盖：

```powershell
docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple api
docker compose up -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:8080/api/health/ready
```

Swagger：<http://127.0.0.1:8080/docs>；Neo4j Browser：<http://127.0.0.1:7474>。

## 本地开发与测试

```powershell
cd python
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -e ".[dev,benchmark]"
.\.venv\Scripts\ruff check .
.\.venv\Scripts\pytest --cov=agents --cov=api --cov=orchestrator --cov=services
```

CI 使用 Fake Chat、Fake Embedding、Fake Vision 和 Fake VL Embedding，不需要真实密钥。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/ingest/upload` | 同步上传单个文档 |
| POST | `/api/ingest/batch` | 最多并发 2 的批量上传 |
| POST | `/api/qa/ask` | `vector / graph / hybrid` 带引用问答 |
| POST | `/api/admin/update` | 按 `doc_id` 生产更新或删除事件 |
| GET | `/api/admin/events/{event_id}` | 查询 CDC 状态 |
| GET/DELETE | `/api/documents/{doc_id}` | 查看或完整删除文档 |
| GET | `/api/admin/stats` | 三个向量集合、图谱、生成模型及 Reranker 用量/延迟/降级统计 |
| GET | `/api/health/live` | 进程存活检查 |
| GET | `/api/health/ready` | SQLite、Chroma、Neo4j、Kafka 和模型配置检查 |

问答 citation 保留 `doc_id/chunk_id/page/sheet/quote/score/type/modality`；响应同时返回
`answerable`、`evidence_score`、分阶段延迟、模型/token 用量和高层 trace。

## 运维与迁移

```powershell
cd python
.\.venv\Scripts\akh-admin preflight
.\.venv\Scripts\akh-admin preflight-multimodal
.\.venv\Scripts\akh-admin migrate-multimodal --dry-run
.\.venv\Scripts\akh-admin migrate-multimodal
.\.venv\Scripts\akh-admin migrate-multimodal --doc-id <doc_id>
.\.venv\Scripts\akh-admin repair
.\.venv\Scripts\akh-admin repair --doc-id <doc_id>
.\.venv\Scripts\akh-admin rebuild <doc_id>
```

迁移命令可断点续跑、逐文档隔离失败并校验 v2 chunk；验证前不会删除旧集合。

## 数据与评测状态

数据下载器、manifest、固定样本选择、查询计划缓存、断点续跑和报告工具均保留；MultiHop-RAG
语料按本轮成本约束使用固定 100 篇文档子集。v2/v3/v4/v5 原始 JSONL、summary 和 gate 报告位于
`python/benchmarks/results/reference/`，不同实验从不互相覆盖。专用命令依次执行真实 Rerank 单请求
预检、分层 10 题预检和分层 100 题门控；任一门槛失败即停止，只有全部通过才续跑剩余 200 题：

```powershell
cd python
.\.venv\Scripts\akh-benchmark dry-run
.\.venv\Scripts\akh-benchmark run-multihop-v5 --confirm-live
.\.venv\Scripts\akh-benchmark run --suite all --fresh-state --no-resume --confirm-live
.\.venv\Scripts\akh-benchmark report
```

当前 100 题门控未通过，程序按设计停止，没有为了得到好看的数字继续付费或调整样本。最终默认关闭
实验性的通用 ComparisonTool；在重新完成同一固定 100 题前，只把它视为工程策略回滚，不声称产生
新的量化提升。

数据来源、许可和校验信息见 `python/benchmarks/manifest.json` 与
`python/benchmarks/manifest.lock.json`。RGB 仅用于 CC BY-NC-SA 4.0 许可范围内的非商业评测。

硅基流动能力说明：[多模态视觉接口](https://docs.siliconflow.cn/cn/userguide/capabilities/multimodal-vision)、
[Embedding 接口](https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings)。
