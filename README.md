# 📚 本地文献语义搜索系统

> **中文提问，0.14秒找到英文论文。完全本地运行，零成本，数据不上传。**

面向**任意学科**研究生的本地 PDF 文献语义搜索工具。
核心特性：**中文查询 → 英文文献匹配**，不需要翻译，不需要想关键词。

---

## 效果演示

```
$ python3 demo_search.py "你的研究主题关键词"

  📚 本地文献语义搜索系统
  索引: 1244 篇  本地 438 + Zotero 806

  🔍 查询:  你的研究主题关键词
  ⚡ 搜索:  0.14 秒
  ✓  找到:  5 篇最相关文献

  1  ████████░░░░  0.717   Most relevant paper title matching your query...
                           2025 · 英文 · Author et al.

  2  ████████░░░░  0.696   Second most relevant paper title...
                           2024 · 英文 · Author B

  3  ████████░░░░  0.686   Third most relevant paper title...
                           2022 · 英文
```

---

## 核心优势

| 对比 | Zotero 内置搜索 | Zotero 插件 (Aria等) | **本系统** |
|------|---------------|---------------------|-----------|
| 搜索范围 | 标题+关键词 | 单篇问答 | **全文语义** |
| 中文查英文 | ❌ | ❌ | **✅** |
| 需要 API Key | ❌ | ✅（付费） | **❌ 零成本** |
| 数据隐私 | 本地 | 上传至第三方 | **完全本地** |
| 跨文献检索 | 有限 | 有限 | **全库同时搜** |
| 文献来源 | Zotero库 | Zotero库 | **电脑任意位置 + Zotero** |

### 文献来源：扫描电脑任意位置

不需要把论文集中到一个地方——脚本会自动扫描：

```
📁 你的本地文献文件夹（任意路径，支持子文件夹）
       ↓ rglob 递归扫描所有 PDF
📚 Zotero 存储目录（自动读取 sqlite 数据库获取元数据）
       ↓ 获取标题、作者、标签、分类
🔀 自动去重（同一论文在两处都有时只保留一份）
       ↓
📊 统一索引（本地 + Zotero，按文件名去重，元数据互补）
```

支持同时配置多个文献来源，新增论文后用 `--incr` 增量更新，无需重建全部索引。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行会自动下载语义模型（约 470MB），之后无需联网

### 2. 修改配置

编辑 `config.py`，填写你的文献目录：

```python
PDF_DIR = Path("~/你的文献目录/")   # PDF 存放位置
ZOTERO_DIR = Path("~/Zotero")       # Zotero 目录（可选）
```

### 3. 建索引（只需运行一次）

```bash
# 第一步：提取 PDF 元数据（约 6 分钟）
python3 build_paper_index.py

# 第二步：生成语义向量（约 10 分钟）
python3 build_embeddings.py
```

### 4. 搜索

```bash
# 中文查询（推荐）
python3 demo_search.py "你的研究主题关键词"

# 中英双语查询（更精准）
python3 demo_search.py "中文查询" --also "English query"

# 纯英文
python3 demo_search.py "your research topic in English"

# 返回更多结果（默认5篇）
python3 search_papers.py "关键词" --top 10
```

---

## 文件说明

```
📁 项目目录（上传 GitHub 的文件）
├── config.py              ← ⚙️ 路径配置（克隆后只需改这里）
├── requirements.txt       ← 依赖列表
├── README.md              ← 本文件
│
├── build_paper_index.py   ← 步骤1：扫描 PDF + Zotero，提取元数据，建 JSON 索引
├── build_embeddings.py    ← 步骤2：计算多语言语义向量，保存 .npz 文件
│
├── search_papers.py       ← 主搜索引擎（完整功能版，支持关键词/语义/混合）
└── demo_search.py         ← 美化输出版（适合截图/演示，彩色框线）

📁 运行后自动生成（加入 .gitignore，不上传）
├── paper_index.json       ← 文献元数据索引（含摘要、关键词、路径）
└── paper_embeddings.npz   ← 语义向量矩阵（N篇 × 384维，约 1~2MB/千篇）
```

### .gitignore 建议

```gitignore
# 生成的索引文件（体积大，且含个人文献路径）
paper_index.json
paper_embeddings.npz
paper_index_readable.md

# 个人配置（含本地路径，不上传）
# config.py  ← 可选：上传模板版，用户自行修改
```

---

## 技术原理

```
PDF 文件
   ↓ PyMuPDF 提取全文
   ↓ jieba 中文分词 + 关键词提取
   ↓ sentence-transformers 多语言模型编码 (384维)
   ↓ 保存为 .npz 向量索引

搜索时：
  中文查询 → 多语言向量编码 → 全库余弦相似度 → Top-K 排序
```

模型：`paraphrase-multilingual-MiniLM-L12-v2`
- 支持 50+ 语言，跨语言语义对齐
- 向量维度：384 维
- 首次下载约 470MB，之后本地缓存

---

## 适用场景

- 有 100~2000 篇本地 PDF 文献的研究生（**任意学科**）
- 需要用中文思路检索英文文献（生态、医学、材料、经济、法学…… 非 CS 方向）
- 希望数据完全本地，不想上传论文到第三方服务
- 用 Zotero 管理文献，想增强搜索能力

---

## 适配其他研究领域

系统的**语义搜索**部分（embedding 向量）完全领域无关，开箱即用。

**关键词搜索**部分包含几个可定制的字典，根据你的领域替换即可：

| 文件 | 变量 | 作用 | 是否需要改 |
|------|------|------|----------|
| `search_papers.py` | `DOMAIN_WORDS` | 告诉分词器识别你的专业术语 | **建议替换** |
| `search_papers.py` | `TOPIC_EXPANSIONS` | 主题→同义词扩展 | **建议替换** |
| `search_papers.py` | `CN_TO_EN_QUERY` | 中文关键词→英文翻译 | **建议补充** |
| `build_paper_index.py` | `DOMAIN_WORDS` | 同上（建索引时用） | **建议替换** |

**定制步骤**（5~10分钟）：

```python
# 1. 在 DOMAIN_WORDS 中填入你领域的专业词汇
DOMAIN_WORDS = [
    "你的专业词1", "你的专业词2", ...  # 让分词器不拆散这些词
]

# 2. 在 TOPIC_EXPANSIONS 中添加你的核心概念
TOPIC_EXPANSIONS = {
    "你的核心概念": ["English synonym1", "synonym2", "中文近义词"],
}

# 3. 在 CN_TO_EN_QUERY 中补充专业词的中英翻译
CN_TO_EN_QUERY = {
    "你的专业词": "your professional term in English",
}
```

> 修改后重新运行 `build_paper_index.py` 即可（embedding 向量无需重建）。

---

## 常见问题

**Q: 建索引需要多久？**
A: 1000 篇 PDF 约 6~10 分钟（只需运行一次，新增文献用 `--incr` 增量更新）

**Q: 内存占用？**
A: 搜索时约 500MB（加载模型），索引文件约 2MB

**Q: 不用 Zotero 可以吗？**
A: 可以，直接把 PDF 放在文件夹里，`config.py` 中 `ZOTERO_DIR` 留空即可

**Q: Windows 可以用吗？**
A: 可以，建议用 WSL 或直接 Python 3.8+

---

## 贡献 / Contributing

欢迎提 Issue 和 PR！特别欢迎：
- 支持更多文献格式（Word、网页、Endnote）
- 添加 Web UI 界面
- 优化中文分词词典

> 如果对你有帮助，欢迎 ⭐ Star！
