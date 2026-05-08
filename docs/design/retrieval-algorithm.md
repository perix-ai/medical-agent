# 医疗文档检索算法设计方案 v0.1

> 目标读者：team 内部
> 状态：草稿（待评审）
> 范围：定义 medical-agent 的索引与检索算法、数据模型、evaluation 框架

---

## 1. 目标与非目标

### 1.1 目标
- 在 ~10 万份中文医疗文档（指南 / 共识 / 临床路径 / RCT 论文 / 教材章节）上构建可用于 **临床诊断辅助** 的检索系统。
- 优化指标顺序：**临床安全性 > 引用可追溯性 > 答案准确率 > 召回率 > 延迟 > 成本**。
- 检索结果必须满足：每条进入 LLM 的临床声明都能精确回标到原文段落（含 doc_id / page / bbox / span）。
- 索引策略层可独立替换、独立评测，便于 ablation。

### 1.2 非目标
- 不追求"开放医疗诊断 100% 准确率"——这是问题性质决定的不可达上限。我们追求 **受限场景（疾病已知 / 决策树封闭）下错答率 < 1% + 不确定时显式 abstain**。
- 不内化知识：LLM 不重写指南，只在原文段落基础上做组合与引用。
- 不做病历存储 / EMR 集成；这些在上游 Agent 层处理。
- 不做实时 streaming 索引；按批处理设计。

### 1.3 设计原则
- **原文不动**：解析产物保留页码、bbox、原始字符 offset，所有 chunk 都能反查原文。
- **分层解耦**：解析 / chunking / 索引 / 检索 / 重排 / 证据聚合 / 回标校验，每层独立模块、独立 metric。
- **结构优先**：先抽出文档的结构化骨架（章节、推荐意见、表格、引用），再做向量化。
- **abstain over hallucination**：检索置信度低于阈值时直接拒答，不让 LLM 兜底。

---

## 2. 总体架构

```
                       ┌──────────────────────────────────────────┐
                       │        Offline / Batch Pipeline           │
                       └──────────────────────────────────────────┘

  raw/*.pdf  ──►  Parser  ──►  Document  ──►  Chunker  ──►  Indexer
                  (MinerU)     (schema.py)    (3-level)     (BM25 +
                                                              dense +
                                                              sparse +
                                                              metadata)
                                                                 │
                                                                 ▼
                                                          Index Stores
                                                          (FAISS / SQLite /
                                                           ES-lite / KG)

                       ┌──────────────────────────────────────────┐
                       │            Online / Query Pipeline        │
                       └──────────────────────────────────────────┘

  user query
       │
       ▼
  Query Rewriter ──► Multi-Route Retrieval ──► RRF Fusion ──► Reranker
  (LLM, 同义词扩展)   (BM25 / dense / sparse                  (cross-encoder)
                       + metadata filter)                          │
                                                                   ▼
                                                          Evidence Aggregator
                                                                   │
                                                                   ▼
                                                            Answer Composer
                                                            (LLM + 强制 citation)
                                                                   │
                                                                   ▼
                                                          Citation Verifier
                                                          (回标失败 → 丢弃声明)
                                                                   │
                                                                   ▼
                                                          final answer + cites
```

---

## 3. 数据模型

所有解析产物落地为 JSONL，便于版本化与 diff。schema 定义在 `src/medical_agent/schema/`，用 pydantic 强约束。

### 3.1 `Document`
一份原始文档（PDF / DOCX / HTML 解析后的根对象）。

```python
class Document(BaseModel):
    doc_id: str                       # 稳定哈希: sha256(source_path + content)[:16]
    source_path: str                  # raw/2022-丙型肝炎防治指南.pdf
    title: str
    authors: list[str] = []
    publisher: str | None             # 中华医学会 / NEJM / ...
    publication_date: date | None
    version: str | None               # "2022年版"
    doc_type: Literal["guideline", "consensus", "rct", "review", "pathway", "textbook"]
    language: str = "zh"
    diseases: list[str] = []          # ICD-10 codes + 中文标签 (后处理填)
    evidence_system: Literal["GRADE", "Oxford", "AHA", "none"] = "none"
    target_population: list[str] = [] # ["成人", "孕妇"] 等
    sections: list[Section]
    tables: list[Table]
    figures: list[Figure]
    recommendations: list[Recommendation]
    references: list[Reference]
    raw_markdown_path: str            # MinerU 输出的 .md 路径
    parse_meta: ParseMeta              # 解析器版本、时间戳、警告
```

### 3.2 `Section`
章节，递归结构。保留 page 范围与字符 span，使任何 chunk 都能回到原文。

```python
class Section(BaseModel):
    section_id: str                   # doc_id + path hash
    path: list[str]                   # ["治疗", "DAA 方案", "基因 1b 型"]
    level: int                        # 1, 2, 3...
    title: str
    text: str                         # 该节正文（不含子节），保留 markdown
    page_start: int
    page_end: int
    char_span: tuple[int, int]        # 在 raw_markdown 中的字符 offset
    children: list[Section] = []
    table_refs: list[str] = []        # 引用的 table_id
    figure_refs: list[str] = []
```

### 3.3 `Recommendation`
**推荐意见** 是医学指南最关键的语义单位，独立成实体。

```python
class Recommendation(BaseModel):
    rec_id: str
    doc_id: str
    section_id: str                   # 所在章节
    number: str | None                # "推荐意见 7"
    statement: str                    # 推荐意见原文 (不改写)
    evidence_grade: str | None        # "A1", "B2", "1A"
    recommendation_strength: str | None  # "强推荐", "弱推荐"
    evidence_system: str | None       # "GRADE" 等
    population: str | None            # "基因 3b 型代偿期肝硬化"
    intervention: str | None          # "索磷布韦/维帕他韦 + 利巴韦林"
    comparator: str | None
    outcome: str | None
    contraindications: list[str] = []
    page: int
    bbox: tuple[float, float, float, float] | None
    surrounding_context: str          # 推荐意见前后各 N 字的论证文本
    cited_refs: list[str] = []        # 引用的文献 ref_id
```

### 3.4 `Table`, `Figure`, `Reference`
- `Table.cells: list[Cell]`，每个 cell 带 row/col header（**不 flatten 成纯文本**）。
- `Figure` 至少存 caption + OCR 文本；流程图可后续接 VLM 抽 caption。
- `Reference.ref_id` 用于 `Recommendation.cited_refs` 反向连接。

### 3.5 `Chunk`
chunking 产物。每个 chunk 保留 parent pointer，便于召回后扩展上下文。

```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    granularity: Literal["doc", "section", "recommendation"]
    parent_chunk_id: str | None
    text: str                         # 用于 embedding 与 BM25 的内容
    display_text: str                 # 给 LLM 的内容（可比 text 长，含必要上下文）
    metadata: ChunkMetadata           # disease, year, evidence_grade, page...
    span_refs: list[SpanRef]          # 回标用的 (doc_id, page, char_span)
    embeddings: dict[str, list[float]] = {}  # 索引时填，runtime 不传
```

---

## 4. 解析层（Parsing）

### 4.1 主解析器：MinerU
- 输入：PDF；输出：Markdown + tables(JSON) + figures(PNG + caption)。
- 选择理由：中文文档 layout / 公式 / 表格综合表现最好；支持 GPU 加速；MIT 协议。
- 集成方式：subprocess 调用 `mineru` CLI 而非 import 库，隔离 Python 依赖冲突；解析产物落 `data/parsed/<doc_id>/`。

### 4.2 fallback：PyMuPDF + 规则
- 当 MinerU 解析失败 / 超时时，退化为 PyMuPDF 抽文本 + camelot 抽表格。
- 规则不追求高质量，只保证 pipeline 不阻塞。失败的文档进入 `data/parse_failures.jsonl` 待人工处理。

### 4.3 后处理（structure extraction）
解析后由 `parsing/post.py` 提取结构化对象：
1. **章节切分**：基于 markdown heading + 中文章节正则（`^第[一二三四五六七八九十]+[章节]`、`^\d+(\.\d+)*\s+`）。
2. **推荐意见抽取**：正则 + LLM 判别。规则覆盖 `推荐意见\s*\d+`、`推荐\s*[:：]`、`Recommendation \d+` 等。证据等级用 `\(([A-D]\d?|[1-5][a-c]?)\)` 抓取后归一化到统一 enum。
3. **表格规范化**：MinerU 输出的 HTML/JSON 表格 → 我方 `Table` 模型；保留 row/col header；跨页表格按 caption 合并。
4. **引用解析**：参考文献节用规则切，每条用 `pyalex` / `crossref` / `regex` 抽 (作者, 年份, 期刊, DOI)。
5. **元数据补全**：`disease`, `target_population` 等用 LLM 在 doc 摘要 + 标题上抽，归一化到 CMeKG（见 §6）。

### 4.4 解析质量门
每份文档输出 `parse_quality.json`：
- `text_coverage`：抽出文本字符数 / PDF 总字符数（PyMuPDF 估算）
- `table_count` / `table_extraction_rate`
- `recommendation_count`
- `heading_consistency`（章节层级是否单调）
- `warnings: list[str]`

低于阈值的文档标 `quality=low` 但仍进索引（带 metadata 标志，检索时降权）。

---

## 5. 多粒度 chunking 策略

这是本期重点验证的策略之一。设计上 **三层 + parent pointer + overlap by structure**，不按字符数硬切。

### 5.1 三层粒度

| 粒度 | 用途 | 字符数（中文） | 何时召回 |
|------|------|---------------|---------|
| `doc` | 路由 / 粗筛 | 200–400（标题+摘要+主疾病） | 第一阶段 routing，决定查哪些文档 |
| `section` | 主检索单位 | 400–1500 | 默认检索粒度 |
| `recommendation` | 强证据精确召回 | 80–300 | 高价值实体；BM25 与 dense 都额外索引 |

### 5.2 切分规则
- **doc**：title + 自动摘要（LLM 在 sections[0:3] 上生成 ≤300 字摘要，避免随机抽段）。
- **section**：以 §3.2 的 `Section` 为单位；若 `text` 过长（>1500 字），按子段落（`\n\n` + 句号）切，**保留 path 前缀作为上下文头**（"治疗 > DAA 方案 > 基因 1b 型：……"），相邻 chunk 重叠 1 句。
- **recommendation**：`statement` + `surrounding_context`（前 N 句论证 + 后 N 句条件说明），强制单独成 chunk。元数据 `is_recommendation=True`，检索时可单路提权。

### 5.3 display_text vs text
- `text`：纯净的检索文本，参与 embedding / BM25。不含 path 前缀以避免噪声。
- `display_text`：给 LLM 看的版本，包含 path 前缀 + parent section 摘要 + 表格引用展开。
- 二者分离让 retrieval 信号干净，LLM 上下文丰富。

### 5.4 表格 chunking（特例）
- 大表（>10 行）按 row 切：每行 + col header → 一个微 chunk，granularity 标 `table_row`（实际归到 `recommendation` 索引同路）。
- 表整体也存一个 `section` 级 chunk，`display_text` 是完整 markdown 表格。
- 这样既能精确召回"基因 3b 型 + 代偿期肝硬化 → 索磷布韦/维帕他韦 + 利巴韦林 12 周"这种 row-level 命中，又能在 LLM 端拿到完整决策表。

### 5.5 反例（明确不做）
- 不做固定 512 / 1024 token 切分。
- 不做语义无关的 sliding window。
- 不在 chunk 边界穿越 `Recommendation` 实体。

---

## 6. 索引层

### 6.1 多路索引设计

| 索引 | 引擎 | 内容 | 用途 |
|------|------|------|------|
| BM25 | SQLite FTS5 起步，1M 文档前迁 ES | `chunk.text` + 医学分词词典 | 术语精确匹配（"HCV 基因 3b 型"） |
| Dense | FAISS（IVF-PQ）→ Milvus | bge-m3 dense (1024 dim) | 语义召回 |
| Sparse | 同 BM25 引擎复用，存 bge-m3 sparse 输出 | bge-m3 sparse | 学习型术语扩展（"HCC↔肝细胞癌"） |
| Metadata | SQLite / Postgres | 全部 ChunkMetadata 字段 | 必经过滤层 |
| KG（可选 v0.2） | NetworkX 起步，量大后 Neo4j | 实体-关系图 | 多跳推理 / disease→guideline 路由 |

### 6.2 中文医学分词
- jieba + 自定义词典（合并 CMeKG / ICD-10 / 药品通用名）
- 路径：`configs/medical_lexicon.txt`，每行 `术语\t权重\t词性`。
- 同义词归一化（"丙肝 / HCV / 丙型病毒性肝炎" → CUI）由独立的 normalizer 模块处理，不污染 BM25 分词。

### 6.3 必须索引的元数据字段（检索时必经 filter）
- `publication_year`（旧版指南诊断时强制排除或显式标注）
- `doc_type` / `evidence_system` / `evidence_grade`
- `diseases`（CMeKG 归一化 ID + 文本）
- `target_population`（成人 / 儿童 / 孕妇 / 老年）
- `version_replaced_by`（旧版指针）

### 6.4 索引构建流程
```
parsed Document  ──► chunking  ──► 并行：
                                      ├─ embed dense (bge-m3 dense)
                                      ├─ embed sparse (bge-m3 sparse)
                                      ├─ build BM25 doc
                                      └─ extract metadata + entities
                                  ──► 落 4 个索引 + 1 张 chunk 主表
```

主表 `chunks` 是真相源，4 个索引都用 `chunk_id` 做 join key。索引可独立重建。

### 6.5 实体归一化
- 工具：`spacy` + CMeKG 词典 + 自训规则。v0.1 不上深度 NER，规则 + 词典覆盖 80% 即可。
- 归一化映射存 `data/entity_map.parquet`，键为 surface form，值为 canonical CUI。

---

## 7. 检索流水线

### 7.1 Query rewriting
- LLM（本地 7B 即可）做三件事：
  1. 口语 → 医学术语（"我得了丙肝" → "HCV / 丙型肝炎"）
  2. 同义词扩展（最多 5 个候选）
  3. 抽出 query 中的实体并归一化（疾病 / 症状 / 药物 / 检查 / 人群条件）
- 输出：`{normalized_query, expansions: [...], entities: [...], required_filters: {...}}`
- Filter 比如"孕妇"出现时强制 `target_population contains 孕妇`。

### 7.2 多路召回
- BM25：`normalized_query + expansions`，top 50
- Dense：`normalized_query`（不扩展，避免漂移），top 50
- Sparse：`normalized_query`，top 50
- KG（可选）：实体节点出发 1 跳邻居，top 20
- **强 metadata filter** 在每路检索前应用（年份、人群、disease 必须命中）。

### 7.3 RRF 融合
经典 Reciprocal Rank Fusion，k=60。融合后 top 30 进 reranker。

### 7.4 Reranker
- bge-reranker-v2-m3，cross-encoder。
- 输入 (query, chunk.text)，输出 score。
- 截断到 top 8（默认值，可调）。
- **若 reranker top1 score < threshold（待标定），整个 query 标 `low_confidence=True`**。

### 7.5 Evidence Aggregation
- 同一 `Recommendation` 的多个 chunk 命中 → 合并为单个证据条目。
- 不同文档对同一 (population, intervention) 的推荐若 **结论冲突**，**保留两个独立证据条目并标 `conflicting=True`**，绝不自动合并。
- 输出：`list[Evidence]`，每个 Evidence 含 statement, grade, sources（多个 chunk）, conflict_with。

### 7.6 Answer composition
- LLM prompt 强制 schema 输出，每条 `claim` 必须带 `evidence_ids: list[chunk_id]`。
- prompt 模板：
  - 系统 prompt 限制不得引入 evidence 之外的临床事实
  - few-shot 示例展示如何说"证据不足，建议……"

### 7.7 Citation Verifier（回标校验）
- 对 LLM 输出的每个 `claim`：
  1. 取声明文本 → embedding
  2. 与该 claim 引用的 evidence 文本算 cosine sim
  3. 若 sim < 0.55（待标定）或 evidence_ids 为空 → **丢弃此 claim**
- 丢弃后若答案为空，整体回退到 `abstain` + 列出 top evidence 让用户自决。
- 这是临床安全的最后一道闸。

### 7.8 Abstain 策略
触发任一条件即 abstain：
- query 改写 LLM 拒识 / 超出医学范畴
- 全部检索路 top1 score 低于阈值
- 检索结果证据系统冲突（指南互相矛盾）
- 引用回标全部失败

abstain 时返回结构化 reason，前端可决定是 escalate 给医生还是要求用户补全信息。

---

## 8. Evidence Chain 协议

### 8.1 SpanRef
每个 chunk 都携带 `span_refs: list[SpanRef]`：

```python
class SpanRef(BaseModel):
    doc_id: str
    page: int
    char_span: tuple[int, int]        # 在 raw_markdown 中的 offset
    bbox: tuple[float, float, float, float] | None  # 在 PDF 中的位置
    quote: str                        # 实际文本（用于回标校验冗余存）
```

### 8.2 引用 ID 规范
- 对外暴露的 citation id 格式：`{doc_id}#sec={section_id}#rec={rec_id?}#p={page}`
- 解析端到端可逆：从 citation id 能反查到原 PDF 页面 + bbox。

### 8.3 输出契约
最终回答必须满足：
```json
{
  "answer": "...",
  "claims": [
    {"text": "...", "evidence_ids": ["...", "..."], "confidence": 0.92}
  ],
  "evidences": [
    {"id": "...", "doc_title": "...", "doc_year": 2022, "quote": "...", "page": 12, "bbox": [...]}
  ],
  "abstain": false,
  "abstain_reason": null,
  "warnings": ["旧版指南 (2019) 已被排除"]
}
```

**前端不允许显示没有 evidences 的 claim。**

---

## 9. 评测框架

### 9.1 评测集分层
- **L0 自动构造**：从指南推荐意见反推 QA（已知答案 → 生成问题），用于回归测试。**不作为发布门**。
- **L1 临床场景集**：医生标注，~200 条，覆盖：
  - 单一指南内决策（"基因 1b 型初治非肝硬化首选方案"）
  - 跨指南检索（"妊娠合并丙肝是否能 DAA 治疗"）
  - 冲突识别（同一问题两份指南推荐不同）
  - abstain 触发（信息不足、超范围）
- **L2 红队集**：医生 + 工程师设计的 adversarial 集，专门测幻觉、跨年份混淆、人群错配。

### 9.2 指标
- **检索侧**：Recall@k（k=5/10/20），nDCG@10，引用覆盖率（answer 中所有 claim 都有 evidence 的比例）。
- **答案侧**：claim accuracy（医生评，每条 claim 对/错/不足以判断），abstain precision / recall，conflict detection F1。
- **安全侧**：错答率（错答 = 有 claim 但与公认指南相反），临床危险错答数（独立计数）。
- **工程侧**：p50 / p95 延迟，索引存储 / 内存占用。

### 9.3 Ablation 维度
每条配置都用同一份 L1 集跑：
1. chunking：单粒度 section vs 多粒度（默认）
2. 检索路：仅 BM25 / 仅 dense / hybrid（默认）
3. reranker on/off
4. citation verifier on/off
5. recommendation 单独索引 on/off
6. metadata filter 严格度

每次提交都打 `eval_report.json`，CI 上回归对比。

### 9.4 标定阈值
- reranker low_confidence threshold
- citation verifier sim threshold
- abstain 触发组合

这些都不应硬编码，统一放 `configs/thresholds.yaml`，每次发布前在 L1 上重新标定。

---

## 10. 目录结构

```
medical-agent/
├── raw/                              # 原始 PDF（用户在维护）
├── data/                             # gitignore
│   ├── parsed/<doc_id>/              # MinerU 输出 + 后处理 JSON
│   ├── chunks/<doc_id>.jsonl
│   ├── entity_map.parquet
│   ├── parse_failures.jsonl
│   └── indices/                      # FAISS / SQLite / ES indices
├── docs/
│   └── design/
│       └── retrieval-algorithm.md    # 本文
├── configs/
│   ├── default.yaml
│   ├── thresholds.yaml
│   └── medical_lexicon.txt
├── src/medical_agent/
│   ├── schema/                       # pydantic models
│   │   ├── document.py
│   │   ├── chunk.py
│   │   └── evidence.py
│   ├── parsing/
│   │   ├── mineru_runner.py
│   │   ├── pymupdf_fallback.py
│   │   ├── post.py                   # section/recommendation/table 抽取
│   │   └── quality.py
│   ├── chunking/
│   │   ├── multi_granularity.py
│   │   └── table.py
│   ├── indexing/
│   │   ├── bm25.py
│   │   ├── dense_faiss.py
│   │   ├── sparse.py
│   │   ├── metadata.py
│   │   └── builder.py                # 编排 4 路索引
│   ├── retrieval/
│   │   ├── query_rewriter.py
│   │   ├── multi_route.py
│   │   ├── rrf.py
│   │   ├── reranker.py
│   │   ├── aggregator.py
│   │   ├── citation_verifier.py
│   │   └── abstain.py
│   ├── eval/
│   │   ├── datasets/                 # L0/L1/L2
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   └── report.py
│   └── cli/
│       ├── ingest.py                 # python -m medical_agent.cli.ingest raw/
│       ├── query.py
│       └── eval.py
├── tests/
│   ├── parsing/
│   ├── chunking/
│   └── retrieval/
├── pyproject.toml
└── README.md
```

---

## 11. 实施路线图

| 阶段 | 范围 | 出口标准 |
|------|------|---------|
| **M0：脚手架** | 目录、pyproject、schema、CLI 占位 | `python -m medical_agent.cli.ingest --help` 可跑 |
| **M1：解析跑通** | MinerU 集成 + post-processing + parse_quality | `raw/` 下两份 PDF 完整解析为 `data/parsed/`；丙肝指南推荐意见数 ≥ 实际数的 90% |
| **M2：多粒度 chunking** | §5 落地 + 单测 | chunk 数 / chunk 长度分布 / 推荐意见 chunk 命中率上 dashboard |
| **M3：单路检索 baseline** | 仅 BM25，单粒度 section | 在 L0 自动 QA 上 Recall@10 报告 |
| **M4：混合检索 + 多粒度** | dense / sparse / RRF / 多粒度合并 | 在 L0 上 Recall@10 ≥ M3 + 10pp |
| **M5：reranker + evidence chain** | citation verifier + abstain | L1 集上引用覆盖率 ≥ 95%，错答率 < 5% |
| **M6：评测闭环 + ablation** | eval CLI + 报告生成 | 每次 PR 跑 ablation，自动评论变化 |
| **M7：扩量** | 加 1k / 10k 文档 | 索引大小、构建时间、p95 延迟符合预算 |

每阶段独立 PR，独立 metric，独立可回滚。

---

## 12. 已识别的风险与待决定事项

### 风险
1. **MinerU 在某些指南上可能 OOM 或表格识别退化**：mitigation 是 fallback 链 + parse_failures 队列。
2. **bge-m3 在医学领域的 zero-shot 表现未知**：M3 后必须在自建集上量化对比 BGE-M3 / Conan-embedding / PubMedBERT-zh。结果决定是否需要 LoRA。
3. **临床冲突检测做不准会反向降低体验**：M5 之前先做"标识冲突，不自动合并"，不做"判定哪个对"。
4. **Citation verifier 的 sim 阈值与 LLM 偏好高度相关**：每次换 LLM 必须重标。
5. **数据合规**：原始指南版权问题。M1 之前确认是否仅限内部使用。

### 待决定（需要业务输入）
- [ ] 检索时默认是否按"最新版优先"还是"全部召回 + 警示"？
- [ ] abstain 后端到端 UX：直接返回 vs 转人工？
- [ ] 是否需要支持非中文文献？（FDA / NEJM 等）
- [ ] LLM 选型：本地 vLLM (Qwen2.5-72B / DeepSeek-V3) vs API。涉及成本与合规。
- [ ] 是否给每条 chunk 标"诊断敏感度"（用药剂量、禁忌、急救方案标 HIGH，背景介绍标 LOW）？检索时高敏度 chunk 的回标更严格。

---

## 13. 下一步

设计文档评审通过后：
1. 落 M0 脚手架（pyproject + schema + 目录）
2. 选 1 份 PDF（建议丙肝指南，结构更规整）做 M1 解析的 reference 实现
3. 同步搭 L0 自动评测集（用解析出的推荐意见反推 QA）
4. M3 baseline 出第一个数字

---

## 附录 A：与开源方案的关系
- **DeepDoc (RAGFlow)**：作为 MinerU 的备选解析器，仅在 §4 解析层借用。
- **Medical-Graph-RAG**：第 6.5 节实体归一化 + 第 7 节多跳检索的设计借鉴其三层图思想。
- **WeKnora Wiki Mode**：明确 **不采用**——医学场景不允许 LLM 改写知识。
- **bge-m3 / bge-reranker-v2-m3**：直接当模型用，不依赖任何 RAG 框架。
