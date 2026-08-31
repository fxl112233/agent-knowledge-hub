# 5 分钟项目演示

## 1. 启动

复制 `python/.env.example` 为 `python/.env`，只在本地填写模型配置。随后在仓库根目录执行：

```powershell
docker compose up -d --build
docker compose ps
```

打开 `http://127.0.0.1:8080/docs` 使用 Swagger。先检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health/live
Invoke-RestMethod http://127.0.0.1:8080/api/health/ready
```

## 2. 上传文档

准备一个含有明确事实的 PDF、DOCX 或图片，例如内容包含“星河项目负责人是张三，交付日期为 2026-03-15”。

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/ingest/upload `
  -F "file=@E:\demo\project.pdf"
```

重点展示响应中的 `doc_id`、版本、chunk/实体/关系数量、`modality_counts` 与视觉降级信息。保存返回的 `doc_id`。

## 3. 带引用问答

```powershell
$body = @{
  question = "星河项目由谁负责，计划何时交付？"
  top_k = 8
  mode = "hybrid"
  document_ids = @("替换为 doc_id")
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/api/qa/ask `
  -ContentType "application/json" `
  -Body $body
```

重点解释：Vector 与 Graph 并行召回，经严格加权 RRF、文档级精排、文档内二次检索和 Chunk 精排后生成答案；citation 中的 `doc_id/chunk_id/page/sheet/quote/modality` 必须对应真实证据。再提一个文档中不存在的问题，展示 `answerable=false`。

## 4. 增量更新

修改上传目录中由系统管理的原文件后，可等待 Watchdog 产生事件；也可调用管理 API：

```powershell
$update = @{ doc_id = "替换为 doc_id"; change_type = "modified" } | ConvertTo-Json
$event = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/api/admin/update `
  -ContentType "application/json" `
  -Body $update

Invoke-RestMethod "http://127.0.0.1:8080/api/admin/events/$($event.event_id)"
```

解释 `event_id` 幂等、added/changed/deleted/unchanged 差异和提交后 SQLite/Chroma/Neo4j 一致性校验。重复同一事件不会重复写入。

## 5. 展示实验而不是背宣传数字

打开 `docs/benchmark-report.md`，只讲三个数字：

- Vector All-evidence Hit@10：45.33%。
- Hybrid All-evidence Hit@10：50.67%，提升 5.34 pp。
- 最终 Answer EM 没有同步提升，所以回滚默认开启的通用 ComparisonTool。

这个结果比“准确率 94%”更有说服力，因为它说明你理解检索指标、生成指标、配对实验和失败分析之间的边界。

## 6. 清理演示数据

```powershell
Invoke-RestMethod -Method Delete `
  "http://127.0.0.1:8080/api/documents/替换为 doc_id"
```

该接口会同步删除 SQLite 目录记录、三个 Chroma 集合中的向量、Neo4j 来源关系和受管上传/图片资产。
