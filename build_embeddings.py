#!/usr/bin/env python3
"""
文献向量索引构建器 - 为语义搜索预计算embedding

依赖: pip install sentence-transformers
模型: paraphrase-multilingual-MiniLM-L12-v2 (~470MB, 支持中英文)

运行方式:
  python3 build_embeddings.py              # 全量构建
  python3 build_embeddings.py --incr       # 增量(只处理新论文)
  python3 build_embeddings.py --model xxx  # 指定模型
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

# 路径从 config.py 读取，config.py 不存在时回退到脚本同目录
try:
    from config import INDEX_PATH, EMBEDDINGS_PATH, EMBEDDING_MODEL as DEFAULT_MODEL
except ImportError:
    _BASE = Path(__file__).parent
    INDEX_PATH = _BASE / "paper_index.json"
    EMBEDDINGS_PATH = _BASE / "paper_embeddings.npz"
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def compose_embedding_text(paper):
    """为每篇论文构建用于embedding的文本
    优先级: Zotero标题 > PDF提取标题 > 文件名
    + 关键词 + 摘要前500字 + Zotero标签/分类
    """
    parts = []

    # 标题（权重高，放前面）
    title = paper.get("zotero_title", "") or paper.get("title_extracted", "") or paper.get("filename", "")
    if title:
        parts.append(title)

    # 关键词
    keywords = paper.get("keywords", "")
    if keywords:
        # 去掉前缀标记
        keywords = keywords.replace("[自动] ", "").replace("[auto] ", "")
        parts.append(keywords)

    # 摘要
    abstract = paper.get("abstract", "")
    if abstract:
        abstract = abstract.replace("[兜底提取] ", "")
        parts.append(abstract[:500])

    # Zotero标签和分类
    tags = paper.get("zotero_tags", [])
    collections = paper.get("zotero_collections", [])
    if tags:
        parts.append(" ".join(tags))
    if collections:
        parts.append(" ".join(collections))

    text = " ".join(parts).strip()

    # 如果没有任何内容，用首页文本兜底
    if len(text) < 20:
        text = paper.get("first_pages_text", "")[:500]

    return text[:1000]  # 限制长度避免模型截断


def build_embeddings(model_name=DEFAULT_MODEL, incremental=False):
    """构建embedding索引"""
    print(f"=== 文献向量索引构建 ===")
    print(f"模型: {model_name}")

    # 加载文献索引
    print(f"\n加载文献索引...")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    papers = index["papers"]
    print(f"  总计 {len(papers)} 篇论文")

    # 增量模式：加载已有embedding
    existing_embeddings = None
    existing_filenames = []
    if incremental and EMBEDDINGS_PATH.exists():
        print(f"  加载已有embedding...")
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        existing_embeddings = data["embeddings"]
        existing_filenames = list(data["filenames"])
        print(f"  已有 {len(existing_filenames)} 篇embedding")

        # 找出需要新计算的
        existing_set = set(existing_filenames)
        new_papers = [p for p in papers if p["filename"] not in existing_set]
        print(f"  需要新计算: {len(new_papers)} 篇")
        if not new_papers:
            print("  无需更新")
            return
    else:
        new_papers = papers

    # 准备文本
    print(f"\n准备embedding文本...")
    texts = []
    filenames = []
    for p in new_papers:
        text = compose_embedding_text(p)
        if text.strip():
            texts.append(text)
            filenames.append(p["filename"])

    print(f"  有效文本: {len(texts)} 篇")
    if not texts:
        print("  无有效文本可embedding")
        return

    # 加载模型
    print(f"\n加载embedding模型 ({model_name})...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"  模型加载耗时: {time.time()-t0:.1f}s")
    print(f"  向量维度: {model.get_sentence_embedding_dimension()}")

    # 计算embedding（分批处理）
    print(f"\n计算embedding...")
    t0 = time.time()
    batch_size = 64
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embeddings)
        if (end % 200 == 0) or end == len(texts):
            elapsed = time.time() - t0
            speed = end / elapsed if elapsed > 0 else 0
            print(f"  进度: {end}/{len(texts)} ({elapsed:.1f}s, {speed:.0f}篇/s)")

    new_embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0
    print(f"  embedding计算完成: {elapsed:.1f}s")

    # 合并增量
    if incremental and existing_embeddings is not None:
        # 移除不再存在的论文
        current_filenames = {p["filename"] for p in papers}
        keep_indices = [i for i, fn in enumerate(existing_filenames) if fn in current_filenames]
        kept_embeddings = existing_embeddings[keep_indices]
        kept_filenames = [existing_filenames[i] for i in keep_indices]

        final_embeddings = np.vstack([kept_embeddings, new_embeddings])
        final_filenames = kept_filenames + filenames
    else:
        final_embeddings = new_embeddings
        final_filenames = filenames

    # 保存
    print(f"\n保存embedding索引...")
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=final_embeddings,
        filenames=np.array(final_filenames, dtype=object),
        model_name=model_name,
        build_date=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    file_size = EMBEDDINGS_PATH.stat().st_size
    print(f"\n=== 构建完成 ===")
    print(f"向量索引: {EMBEDDINGS_PATH} ({file_size/1024/1024:.1f} MB)")
    print(f"论文数: {len(final_filenames)}")
    print(f"向量维度: {final_embeddings.shape[1]}")
    print(f"模型: {model_name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    incremental = "--incr" in args
    model_name = DEFAULT_MODEL

    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model_name = args[i + 1]

    build_embeddings(model_name=model_name, incremental=incremental)
