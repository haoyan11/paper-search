"""
配置文件 — 克隆项目后只需修改这里
"""
from pathlib import Path

# ============================================================
# 必须修改：你的 PDF 文献存放目录（支持子文件夹递归扫描）
# 示例 Windows: Path("C:/Users/你的用户名/Documents/论文")
# 示例 Mac/Linux: Path("~/Documents/论文")
# ============================================================
PDF_DIR = Path("~/论文")           # ← 改成你的文献目录

# ============================================================
# 可选：Zotero 数据目录（留空字符串则跳过 Zotero 扫描）
# Windows: Path("C:/Users/你的用户名/Zotero")
# Mac:     Path("~/Zotero")
# Linux:   Path("~/Zotero")  或  Path("~/snap/zotero-snap/common/Zotero")
# ============================================================
ZOTERO_DIR = Path("~/Zotero")      # ← 改成你的 Zotero 路径，不用则填 Path("")

# ============================================================
# 索引文件存放位置（默认放在脚本同目录，一般不需要改）
# ============================================================
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "paper_index.json"
EMBEDDINGS_PATH = BASE_DIR / "paper_embeddings.npz"

# ============================================================
# 语义模型（首次运行自动下载 ~470MB，之后本地缓存）
# ============================================================
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
