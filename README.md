# 企业制度知识助手（v1.0）

一个可运行的完整 Agent：回答企业制度问题并注明条款出处，支持多轮对话；提交报销申请时，金额超过阈值自动进入人工审批；全程有权限控制、注入防御、审计与可观测性。

## 目录

- [快速开始](#快速开始)
- [运行模式与配置](#运行模式与配置)
- [使用流程](#使用流程)
- [HTTP 接口](#http-接口)
- [项目结构](#项目结构)
- [验证与评测](#验证与评测)
- [Docker 部署](#docker-部署)
- [设计与安全要点](#设计与安全要点)
- [故障排查](#故障排查)

---

## 快速开始

需要 Python 3.10+。在项目根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# 1. 将 docs/ 下的制度文档切分、向量化并入库
python -m app.rag.ingest

# 2. 启动 API
uvicorn app.api:app --reload

# 3. 打开审批台（含对话测试）
# http://127.0.0.1:8000/admin/approvals
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","llm_mode":"local","indexed_chunks":14}
```

`llm_mode` 为 `local` 表示正在使用内置规则模型与哈希向量；为 `openai` 表示远端模型可用。

---

## 运行模式与配置

项目**无需任何密钥即可运行**：未配置 `OPENAI_API_KEY` 时使用本地规则模型、哈希向量、内存 checkpoint 与本地文件索引。若配置了远端模型但服务端不支持指定模型，启动时会自动探测并降级到本地模式，健康检查会显示实际使用的模式。

> 当前代码通过环境变量读取配置，不会自动加载 `.env` 文件。`.env.example` 是变量清单；本地运行请用 `export`（Linux/macOS）或 `$env:NAME="value"`（PowerShell）设置变量。Docker Compose 会自动注入其中声明的变量。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | 留空启用本地降级模式 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容接口地址 |
| `MODEL_NAME` | `gpt-4o-mini` | 工具调用模型 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 向量模型 |
| `EMBEDDING_DIM` | 自动 | 显式覆盖向量维度（远端 1536 / 本地 256） |
| `DATABASE_URL` | 空 | 配置后启用 PostgresSaver + pgvector；留空用内存 checkpoint + 本地索引 |
| `DOCS_DIR` | `docs` | 制度文档目录 |
| `DATA_DIR` | `data` | 索引、审批单、审计与业务记录目录 |
| `APPROVAL_THRESHOLD` | `2000` | 报销人工审批阈值（元） |
| `TOKEN_BUDGET` | `8000` | 每个 thread 的 token 预算 |
| `MAX_TURNS` | `6` | 每轮对话最多工具调用轮次 |
| `RETRIEVAL_TOP_K` | `4` | 检索返回条数 |
| `PROMPT_TOP_K` | `2` | 注入提示词的条款数（成本裁剪） |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 空 | 配置后上传 tracing，否则仅写本地 span 日志 |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse 地址 |
| `API_PORT` | `8000` | `python -m app.api` 启动时的端口 |

---

## 使用流程

### 1. 制度问答

以 `query` 角色发送消息，SSE 返回 `message` 事件，`citations` 中带条款号、版本和生效日期。

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"t1","role":"query","message":"一线城市住宿费上限是多少？"}'
```

回答中的制度内容来自 `<untrusted_doc>` 包装的检索结果；文档内出现的指令会被忽略，无出处的问题回答“制度未提及”。

### 2. 提交报销（人工审批）

以 `write` 角色提交大额报销：

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"t2","role":"write","message":"我要报销住宿费2500元，类别差旅，事由出差住宿"}'
```

步骤：

1. SSE 先收到 `interrupt` 事件，其中包含审批单 `id`、`thread_id`、金额与原因；
2. 在审批台点“批准/拒绝”，或直接调用续跑接口：

```bash
curl -N -X POST http://127.0.0.1:8000/threads/t2/resume \
  -H "Content-Type: application/json" \
  -d '{"approved":true,"approver":"admin","role":"write"}'
```

3. 批准后续跑返回“报销申请已提交”及单号；拒绝则不会产生业务记录。

审批单状态可在 `/api/approvals` 或审批台查看。金额不超过阈值时跳过审批，直接提交。

### 3. 角色权限

| 工具 | `query` | `write` | `admin` |
| --- | --- | --- | --- |
| `search_policy` | 允许 | 允许 | 允许 |
| `read_doc` | 允许 | 允许 | 允许 |
| `submit_reimbursement` | 拒绝 | 允许（超阈值需审批） | 允许 |
| `send_notification` | 拒绝 | 拒绝 | 允许 |

越权调用会被拦截并写入审计日志。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查、模型模式、索引条数 |
| `POST` | `/chat` | SSE 对话。Body：`{thread_id, message, role, user_id}` |
| `POST` | `/threads/{thread_id}/resume` | 审批后续跑。Body：`{approved, approver, role}` |
| `GET` | `/api/approvals` | 全部审批单（已脱敏） |
| `POST` | `/api/approvals/{approval_id}/decide` | 仅更新审批单状态；实际续跑仍需调用 resume 接口 |
| `GET` | `/admin/approvals` | 审批管理页（含对话测试） |
| `POST` | `/admin/ingest` | 重建索引，返回 `{"chunks": N}` |

SSE 事件类型：

| event | 说明 |
| --- | --- |
| `message` | 最终答复，含 `content` 与 `citations` |
| `interrupt` | 需要人工审批，含审批单信息 |
| `done` | 本轮结束 |
| `error` | 执行异常，消息已脱敏 |

---

## 项目结构

```text
policy-assistant/
├── app/
│   ├── api.py              # FastAPI + SSE、审批续跑、运维接口
│   ├── graph.py            # LangGraph 状态图与 checkpoint 选择
│   ├── nodes.py            # agent / retrieve / tools / approval 节点
│   ├── tools.py            # 工具实现与 invoke_tool 统一入口
│   ├── security.py         # 权限矩阵、校验、脱敏与审计
│   ├── approvals.py        # 审批单持久化
│   ├── observability.py    # Langfuse / 本地 span、预算闸门
│   ├── models.py           # OpenAI 兼容模型与本地降级实现
│   ├── config.py           # 环境变量配置
│   └── rag/
│       ├── ingest.py       # 文档解析与按条款切分
│       ├── retriever.py    # BM25 + 向量 RRF 混合检索
│       └── store.py        # 本地索引 / pgvector
├── docs/sample/            # 示例制度文档（Markdown）
├── evals/                  # 20 条评测用例与评测脚本
├── tests/                  # 单元测试
├── web/index.html          # 审批管理页
├── data/                   # 运行时生成：索引、审批、审计、业务记录
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 验证与评测

```bash
# 单元测试：安全、RAG、审批图、预算闸门
pytest

# 回归评测：10 条标准问法 + 5 条口语改写 + 3 条陷阱问题 + 2 条注入攻击
python -m evals.run_eval
```

评测脚本在索引为空时会自动入库；全部通过返回退出码 0，可用于 CI。

---

## Docker 部署

```bash
docker compose up -d --build
```

包含服务：

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| `api` | `8000` | FastAPI 服务 |
| `postgres` | `5432` | pgvector：checkpoint 与向量索引 |
| `langfuse` | `3000` | Tracing 服务 |
| `langfuse-db` | 内部 | Langfuse 独立数据库 |

如需接入远端模型，在启动前设置 `OPENAI_API_KEY` 等变量；留空则容器同样以本地模式运行。生产环境请替换 `docker-compose.yml` 中的 Langfuse 演示密钥。

---

## 设计与安全要点

- **编排**：`agent` 决定调用工具或作答；`retrieve` 执行 RAG；`tools` 执行普通工具；`approval` 在敏感工具前 `interrupt()`。
- **唯一入口**：所有工具经 `invoke_tool`，顺序为权限过滤 → 参数校验 → 审批门 → 执行 → 审计。
- **RAG**：按条款切分（300 字 + 50 重叠），混合检索后 RRF 合并、rerank，回答必须引用条款号。
- **提示注入防御**：制度内容统一包裹在 `<untrusted_doc>` 中，系统提示声明其中指令一律忽略；陷阱问题不作答。
- **审计与脱敏**：密钥、身份证号、手机号、银行卡号在日志中脱敏；审计记录包含用户、时间、工具、参数与审批结果。
- **降级**：检索无结果时回答“制度未提及”，检索异常按空结果处理并打印警告；远端模型不可用自动切本地；超预算或超轮次熔断并友好提示。

---

## 故障排查

| 现象 | 处理 |
| --- | --- |
| 回答“制度未提及”但知识库应有内容 | 检查文档是否在 `DOCS_DIR`，执行 `POST /admin/ingest` 重建索引 |
| `/health` 显示 `llm_mode=local` 但已配置密钥 | 远端模型或 embedding 模型不被服务端支持，已自动降级；检查启动日志中的 `[warn]` |
| 对话停在待审批 | 调用 `/api/approvals` 查看待办，再通过 `/threads/{thread_id}/resume` 批准或拒绝 |
| 想清空运行时数据 | 停止服务后删除 `data/` 目录，再执行 `python -m app.rag.ingest` |
| 中文问法检索不准 | 确认已入库最新版本；可在 `app/rag/retriever.py` 的 `SYNONYM_EXPANSION` 补充口语同义词 |
