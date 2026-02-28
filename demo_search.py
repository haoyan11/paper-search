#!/usr/bin/env python3
"""
文献语义搜索 - 截图演示脚本
用法: python3 demo_search.py "中文查询" ["--also" "English query"]
功能: 美化输出，适合截图/演示
"""

import sys
import os
import time

# 在导入任何会产生进度条的库之前，重定向 stderr 以屏蔽 tqdm/transformers 加载信息
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

# 预加载会产生噪声输出的库
try:
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    pass

# 恢复 stderr（rich 需要用到）
sys.stderr = _original_stderr

# ===========================
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# ========= 路径配置 =========
BASE = Path(__file__).parent
INDEX_PATH = BASE / "paper_index.json"
EMBEDDINGS_PATH = BASE / "paper_embeddings.npz"

console = Console(width=100, highlight=False)

# ========= 加载数据 =========
def load_index():
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    papers = list(data["papers"].values()) if isinstance(data.get("papers"), dict) else data.get("papers", [])
    stats = data.get("stats", {})
    return papers, stats

def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return None, None, None
    d = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return d["embeddings"], list(d["filenames"]), str(d.get("model_name", "unknown"))

# ========= 语义搜索 =========
def get_model(model_name):
    """静默加载模型"""
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    try:
        model = SentenceTransformer(model_name)
    finally:
        sys.stderr = old_stderr
    return model

def semantic_search(query, papers, embeddings, filenames, model, top_n=5):
    """返回 (paper, score) 列表"""
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm = embeddings / norms
    sims = emb_norm @ q_norm

    # 建立文件名→paper映射
    fname_to_paper = {p["filename"]: p for p in papers}
    results = []
    top_idx = np.argsort(sims)[::-1][:top_n * 3]
    for idx in top_idx:
        fname = filenames[idx]
        p = fname_to_paper.get(fname)
        if p:
            results.append((p, float(sims[idx])))
        if len(results) >= top_n:
            break
    return results

# ========= 美化输出 =========
def make_score_bar(score, width=12):
    """将相似度分数转换为可视化进度条"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar

def format_title(p):
    """提取最佳显示标题"""
    title = p.get("zotero_title") or p.get("title_extracted") or p["filename"]
    # 去掉文件扩展名
    if title.endswith(".pdf"):
        title = title[:-4]
    # 截断过长标题
    if len(title) > 80:
        title = title[:77] + "..."
    return title

def format_authors(p):
    authors = p.get("zotero_authors", [])
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    elif len(authors) <= 3:
        return ", ".join(authors)
    else:
        return f"{authors[0]} 等 ({len(authors)}人)"

def print_demo_results(query, results, elapsed, stats):
    """美化打印搜索结果"""

    # ===== 标题面板 =====
    header_text = Text()
    header_text.append("📚 本地文献语义搜索系统\n", style="bold white")
    header_text.append(f"  索引: {stats.get('total_papers', '?')} 篇论文  ", style="dim white")
    header_text.append(f"本地 {stats.get('local_papers', '?')} + Zotero {stats.get('zotero_papers', '?')}  ", style="dim cyan")
    header_text.append(f"中文 {stats.get('chinese_papers', '?')} / 英文 {stats.get('english_papers', '?')}", style="dim green")

    console.print(Panel(
        header_text,
        border_style="bright_blue",
        padding=(0, 2),
    ))

    # ===== 查询信息 =====
    console.print()
    console.print(f"  [bold yellow]🔍 查询:[/]  [bold white]{query}[/]")
    console.print(f"  [bold cyan]⚡ 搜索:[/]  [bold white]{elapsed:.2f} 秒[/]  [dim](向量计算 + 相似度排序)[/]")
    console.print(f"  [bold green]✓  找到:[/]  [bold white]{len(results)} 篇最相关文献[/]")
    console.print()
    console.print(Rule("[dim]搜索结果[/]", style="bright_blue"))
    console.print()

    # ===== 结果列表 =====
    for rank, (p, score) in enumerate(results, 1):
        # 相关度颜色
        if score >= 0.7:
            score_color = "bright_green"
        elif score >= 0.5:
            score_color = "yellow"
        else:
            score_color = "white"

        title = format_title(p)
        authors = format_authors(p)
        year = p.get("year") or "年份未知"
        lang = "🇨🇳 中文" if p.get("language") == "zh" else "🇺🇸 英文"
        is_thesis = "🎓 学位论文" if p.get("is_thesis") else ""
        source_tag = "[cyan]Z[/cyan]" if p.get("source") == "zotero" else "[green]L[/green]"
        bar = make_score_bar(score)

        # 构建条目
        rank_text = f"[bold white on bright_blue] {rank} [/]"
        console.print(
            f"  {rank_text}  [{score_color}]{bar}[/]  [{score_color}]{score:.3f}[/]  "
            f"{source_tag}"
        )
        console.print(f"     [bold white]{title}[/]")

        meta_parts = [f"[dim]{year}[/]", f"[dim]{lang}[/]"]
        if is_thesis:
            meta_parts.append(f"[dim yellow]{is_thesis}[/]")
        if authors:
            meta_parts.append(f"[dim]{authors}[/]")
        console.print("     " + "  |  ".join(meta_parts))

        # 摘要片段
        abstract = p.get("abstract", "")
        if abstract:
            snippet = abstract[:160].replace("\n", " ").strip()
            if len(abstract) > 160:
                snippet += "..."
            console.print(f"     [dim italic]{snippet}[/]")

        console.print()

    # ===== 底部提示 =====
    console.print(Rule(style="dim"))
    console.print(
        f"  [dim]🔧 系统：1244篇论文 · 多语言语义向量 · 中文查询→英文文献匹配[/]"
    )
    console.print(
        f"  [dim]📂 模型：paraphrase-multilingual-MiniLM-L12-v2 (384维)[/]"
    )
    console.print()


def main():
    args = sys.argv[1:]
    if not args:
        console.print("[red]用法: python3 demo_search.py \"搜索查询\" [--also \"额外查询\"][/]")
        console.print("[dim]示例: python3 demo_search.py \"植被物候对蒸散发的影响\" --also \"vegetation phenology evapotranspiration\"[/]")
        return

    # 解析参数
    query_parts = []
    also_parts = []
    i = 0
    while i < len(args):
        if args[i] == "--also" and i + 1 < len(args):
            also_parts.append(args[i + 1])
            i += 2
        elif args[i].startswith("--"):
            i += 1
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    also_queries = also_parts

    # 合并所有查询
    all_queries = [query] + also_queries
    full_query = " ".join(all_queries)

    # 加载数据（静默）
    papers, stats = load_index()
    embeddings, filenames, model_name = load_embeddings()

    if embeddings is None:
        console.print("[red]⚠ 未找到向量索引，请先运行 build_embeddings.py[/]")
        return

    # 加载模型（不计入搜索时间，静默加载）
    model = get_model("paraphrase-multilingual-MiniLM-L12-v2")

    # 搜索（仅计向量编码+相似度计算时间）
    t0 = time.time()
    results = semantic_search(full_query, papers, embeddings, filenames, model, top_n=5)
    elapsed = time.time() - t0

    # 打印结果
    display_query = query
    if also_queries:
        display_query += f" + {' '.join(also_queries)}"
    print_demo_results(display_query, results, elapsed, stats)


if __name__ == "__main__":
    main()
