#!/usr/bin/env python3
"""
文献搜索引擎 v3 - jieba分词 + 语义搜索 + 混合排序

用法:
  python3 search_papers.py "研究主题关键词"              # 混合搜索(默认)
  python3 search_papers.py --keyword "关键词1 关键词2"   # 仅关键词搜索
  python3 search_papers.py --semantic "研究问题描述"     # 仅语义搜索
  python3 search_papers.py --topic "核心研究问题"        # 主题搜索（自动扩展）
  python3 search_papers.py --folder "子文件夹名"         # 按文件夹/分类筛选
  python3 search_papers.py --year-sort "关键词"          # 按年份排序
  python3 search_papers.py --similar "作者名"            # 相似论文推荐
  python3 search_papers.py --stats                       # 显示索引统计
  python3 search_papers.py --top 20 "关键词"             # 返回更多结果
  python3 search_papers.py "中文查询" --also "English query"  # 多查询融合
"""

import json
import re
import sys
import os
import time
import jieba
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# 路径从 config.py 读取，config.py 不存在时回退到脚本同目录
try:
    from config import INDEX_PATH, EMBEDDINGS_PATH
except ImportError:
    _BASE = Path(__file__).parent
    INDEX_PATH = _BASE / "paper_index.json"
    EMBEDDINGS_PATH = _BASE / "paper_embeddings.npz"

# ============= jieba 领域词典（与build_paper_index.py保持一致） =============
# 请填入你研究领域的专业词汇，让分词器正确识别这些术语（不会被拆散）
#
# ⚡ 推荐：让 Claude / ChatGPT 帮你一键生成（见 README "使用AI助手定制"章节）
#    提示词示例："我是[你的研究方向]方向研究生，帮我生成jieba领域词典的词汇列表"
#
# 手动填写示例：
#   医  学：["心肌梗死", "动脉粥样硬化", "高血压", "胰岛素抵抗", "靶向治疗"]
#   计算机：["神经网络", "注意力机制", "大语言模型", "迁移学习", "强化学习"]
#   材料学：["石墨烯", "纳米管", "超导体", "钙钛矿", "金属有机框架"]
#   经济学：["货币政策", "供应链管理", "价格指数", "市场失灵", "宏观经济"]
DOMAIN_WORDS = [
    # 在这里填入你领域的专业词汇
    # "专业词汇1", "专业词汇2", "专业词汇3",
]
for w in DOMAIN_WORDS:
    jieba.add_word(w)

STOPWORDS_ZH = {
    '的', '了', '在', '是', '和', '与', '对', '及', '等', '为', '中',
    '上', '下', '有', '无', '不', '也', '又', '被', '或', '将', '把',
    '从', '到', '以', '用', '可', '能', '会', '要', '就', '都', '而',
    '但', '这', '那', '其', '之', '所', '者', '此', '个', '已', '由',
    '于', '则', '并', '且', '如', '进行', '通过', '利用', '采用', '分析',
    '研究', '结果', '表明', '显示', '提出', '提高', '基于', '方法',
    '影响', '变化', '条件', '不同', '情况', '关系', '作用', '具有',
    '相关', '较大', '较小', '明显', '主要', '一定', '同时', '以及',
    '大学', '学院', '学报', '教授', '博士', '硕士', '导师', '作者',
    '北京', '上海', '南京', '中国', '工程', '学位', '论文', '专业',
    '科学', '科学院', '研究所', '研究院', '实验室', '中心',
    '中文', '英文', '翻译', '全文', '摘要', '关键', '参考', '文献',
}

# 主题→同义词扩展映射
# ⚡ 推荐：让 Claude / ChatGPT 帮你生成（见 README "使用AI助手定制"章节）
# 格式："中文核心概念": ["English synonym1", "synonym2", "中文同义词", ...]
TOPIC_EXPANSIONS = {
    # 以下为示例条目（可替换为你的领域）
    "医学影像": ["medical imaging", "MRI", "CT scan", "X-ray", "ultrasound",
                "radiology", "image segmentation", "computer-aided diagnosis", "诊断"],
    "大语言模型": ["large language model", "LLM", "GPT", "BERT", "transformer",
                  "ChatGPT", "fine-tuning", "prompt engineering", "自然语言处理", "NLP"],
    "药物发现": ["drug discovery", "drug design", "molecular docking", "target",
               "靶点", "抗体", "小分子", "临床试验", "pharmacology"],
    # 在此继续添加你的核心研究概念...
    # "你的主题": ["English synonym1", "synonym2", "中文同义词"],
}

# ============= 中文→英文查询翻译（语义搜索用） =============
# 完整查询模板（优先匹配，最精准）
# ⚡ 推荐：让 AI 帮你生成你领域最常用的查询→英文扩展对
# 格式："中文查询短语": "English keyword expansion"
_QUERY_TEMPLATES = {
    # 示例（医学/计算机，请替换为你自己的）：
    # "靶向治疗耐药性机制": "targeted therapy drug resistance mechanism cancer",
    # "大模型推理能力评估": "large language model reasoning benchmark evaluation",
    # "医学图像分割深度学习": "medical image segmentation deep learning CNN",
}

# 词级翻译（用于英文关键词搜索通道，让中文查询能匹配英文文献）
# ⚠️ 此字典必须根据你的研究领域定制，否则中文查询无法命中英文文献！
CN_TO_EN_QUERY = {
    # ======= 通用学术动词/方法（各学科均适用，建议保留）=======
    "分类": "classification",
    "识别": "identification detection recognition",
    "影响": "effect impact influence",
    "机制": "mechanism pathway",
    "趋势": "trend change",
    "响应": "response effect",
    "模拟": "simulation modeling",
    "预测": "prediction forecast",
    "评估": "assessment evaluation",
    "归因": "attribution",

    # ======= ⚠️ 请在此填入你领域的专用词对（非常重要！）=======
    # 语义搜索（embedding）无需此字典即可工作；
    # 但 BM25 关键词通道依赖该词典将中文词翻译为英文关键词，
    # 缺失时中文查询将无法命中仅有英文标题/摘要的论文。
    # 词对越齐全，"中文查询 → 英文文献"的召回效果越好。
    # ⚡ 推荐：让 Claude / ChatGPT 帮你30秒生成30~50个词对（见 README）
    #
    # 示例（医学，可删除并替换为你的领域）：
    # "心肌梗死": "myocardial infarction heart attack",
    # "靶向治疗": "targeted therapy inhibitor kinase",
    # "免疫治疗": "immunotherapy checkpoint PD-1 PD-L1",
    # "临床试验": "clinical trial randomized controlled",
    # "生物标志物": "biomarker marker",
    #
    # 示例（计算机，可删除并替换为你的领域）：
    # "大语言模型": "large language model LLM GPT transformer",
    # "图像识别": "image recognition classification CNN",
    # "强化学习": "reinforcement learning reward policy",
}

# ============= 英→中标签映射（给英文论文生成中文关键词） =============
# ⚠️ 此字典必须根据你的研究领域定制！填写越多，中文检索英文论文的覆盖率越高。
_EN_TO_CN_TAGS = {
    # ======= 通用学术方法标签（各学科均适用，建议保留）=======
    "attribution": "归因分析",
    "trend": "趋势",
    "model": "模型",
    "classification": "分类",

    # ======= ⚠️ 请在此添加你领域的专用英→中标签（非常重要！）=======
    # 这些标签用于为英文论文自动生成中文关键词，让英文文献可以被中文词汇检索到。
    # 填写越齐全，中文关键词搜索英文论文的覆盖率越高。
    # ⚡ 推荐：让 Claude / ChatGPT 帮你生成（见 README）
    #
    # 示例（医学，可删除并替换为你的领域）：
    # "myocardial infarction": "心肌梗死",
    # "targeted therapy": "靶向治疗",
    # "immunotherapy": "免疫治疗",
    # "clinical trial": "临床试验",
    # "biomarker": "生物标志物",
    # "drug resistance": "耐药性",
    # "tumor microenvironment": "肿瘤微环境",
    #
    # 示例（计算机，可删除并替换为你的领域）：
    # "large language model": "大语言模型",
    # "image segmentation": "图像分割",
    # "reinforcement learning": "强化学习",
    # "knowledge graph": "知识图谱",
    # "object detection": "目标检测",
}
_COMPOUND_TAG_RULES = [
    # ⚡ 让 AI 帮你生成（见 README "使用AI助手定制"章节）
    # 格式：({"主题标签A", "主题标签B"}, "复合主题标签")
    # 示例（医学/计算机，请替换）：
    # ({"深度学习", "医学影像"}, "医学影像深度学习"),
    # ({"大语言模型", "推理"}, "大模型推理能力"),
]

def _generate_cn_topics(paper):
    """为英文论文生成中文主题标签"""
    parts = [paper.get('keywords', ''), paper.get('abstract', ''), paper.get('title_extracted', '')]
    text = ' '.join(p for p in parts if p).lower()
    # 若核心元数据不足（<100字符），补充first_pages
    if len(text) < 100:
        fp = paper.get('first_pages', '')
        if fp:
            text += ' ' + fp[:2000].lower()
    cn = set()
    for en in sorted(_EN_TO_CN_TAGS.keys(), key=len, reverse=True):
        if en in text:
            cn.add(_EN_TO_CN_TAGS[en])
    # 也检查folder名称中的中文关键词
    folder = paper.get('folder', '')
    if folder:
        # 从文件夹名称中提取中文关键词（_EN_TO_CN_TAGS中的中文值 + 你添加的领域词）
        for kw in set(_EN_TO_CN_TAGS.values()):
            if kw in folder:
                cn.add(kw)
    for conds, tag in _COMPOUND_TAG_RULES:
        if conds.issubset(cn):
            cn.add(tag)
    return ' '.join(cn)

def _translate_query_wordlevel(query):
    """词级翻译：不用模板，仅做最长匹配词翻译（用于英文关键词搜索通道）"""
    q_norm = re.sub(r'[\s，。、：；？！\u201c\u201d\u2018\u2019（）()的与和对在于中]+', '', query)
    total_cn_chars = len(re.findall(r'[\u4e00-\u9fff]', q_norm))
    text = q_norm
    parts = []
    seen = set()
    translated_chars = 0
    sorted_keys = sorted(CN_TO_EN_QUERY.keys(), key=len, reverse=True)
    while text:
        matched = False
        for key in sorted_keys:
            if text.startswith(key):
                en = CN_TO_EN_QUERY[key]
                if en not in seen:
                    parts.append(en)
                    seen.add(en)
                translated_chars += len(key)
                text = text[len(key):]
                matched = True
                break
        if not matched:
            m = re.match(r'[a-zA-Z]+', text)
            if m:
                w = m.group()
                if w not in seen:
                    parts.append(w)
                    seen.add(w)
                text = text[len(w):]
            else:
                text = text[1:]
    # 若翻译覆盖率<50%（大量中文地名/专有名词未翻译），返回空以避免泛化匹配
    if total_cn_chars > 0 and translated_chars / total_cn_chars < 0.5:
        return ''
    return ' '.join(parts)

def _translate_query(query):
    """将中文查询翻译为英文（用于语义搜索）"""
    # 1) 先尝试完整查询模板
    q_norm = re.sub(r'[\s，。、：；？！\u201c\u201d\u2018\u2019（）()]+', '', query)
    for cn, en in _QUERY_TEMPLATES.items():
        if q_norm == re.sub(r'\s+', '', cn):
            return en
    for cn, en in sorted(_QUERY_TEMPLATES.items(), key=lambda x: len(x[0]), reverse=True):
        cn_norm = re.sub(r'\s+', '', cn)
        if q_norm in cn_norm:
            return en
    # 2) 最长匹配优先（解决jieba切词不匹配CN_TO_EN_QUERY键的问题）
    text = q_norm
    parts = []
    seen = set()
    sorted_keys = sorted(CN_TO_EN_QUERY.keys(), key=len, reverse=True)
    while text:
        matched = False
        for key in sorted_keys:
            if text.startswith(key):
                en = CN_TO_EN_QUERY[key]
                if en not in seen:
                    parts.append(en)
                    seen.add(en)
                text = text[len(key):]
                matched = True
                break
        if not matched:
            # 检查英文字符
            m = re.match(r'[a-zA-Z]+', text)
            if m:
                w = m.group()
                if w not in seen:
                    parts.append(w)
                    seen.add(w)
                text = text[len(w):]
            else:
                text = text[1:]  # 跳过无法匹配的字符
    return ' '.join(parts)


# ============= v3: jieba分词 =============
def tokenize(text):
    """分词（中英文混合）- v3使用jieba切词"""
    tokens = set()

    # 英文单词
    en_words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{1,}', text)
    tokens.update(w.lower() for w in en_words if len(w) >= 2)

    # 中文：jieba搜索模式分词
    zh_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', text))
    if zh_text:
        words = jieba.cut_for_search(zh_text)
        tokens.update(w for w in words if len(w) >= 2 and w not in STOPWORDS_ZH)

    return tokens


def parse_query(query):
    """解析查询：使用jieba分词提取中英文查询词"""
    terms = []

    # 英文单词
    en_words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]+', query)
    terms.extend(en_words)

    # 中文部分：jieba分词
    zh_parts = re.findall(r'[\u4e00-\u9fff]+', query)
    for zh in zh_parts:
        words = jieba.cut(zh)
        terms.extend(w for w in words if len(w) >= 2 and w not in STOPWORDS_ZH)

    return terms


def expand_query(query_terms):
    """扩展查询词（添加同义词/相关词）"""
    expanded = set(t.lower() for t in query_terms)
    expanded.update(query_terms)  # 保留原始大小写
    matched_topics = []

    for term in query_terms:
        term_lower = term.lower()
        for topic, synonyms in TOPIC_EXPANSIONS.items():
            synonyms_lower = [s.lower() for s in synonyms]
            if term_lower in synonyms_lower or term in topic:
                expanded.update(s.lower() for s in synonyms)
                if topic not in matched_topics:
                    matched_topics.append(topic)

    return expanded, matched_topics


# ============= 关键词搜索 =============

def score_paper_keyword(paper, query_tokens, expanded_tokens):
    """计算论文与查询的关键词相关性得分"""
    score = 0.0

    fields = {
        "filename": 3.0,
        "keywords": 5.0,
        "abstract": 4.0,
        "title_extracted": 3.5,
        "first_pages_text": 1.0,
        "folder": 2.0,
        "zotero_meta": 2.5,  # v3: Zotero元数据
        "cn_topics": 3.0,    # v4: 英文论文的中文标签（适度权重，避免过度压制中文论文）
    }

    matched_fields = []
    matched_terms = set()
    precomputed = paper.get("tokens", {})

    for field, weight in fields.items():
        # 优先使用预计算tokens
        if field in precomputed:
            text_tokens = set(precomputed[field])
        elif field == "first_pages_text":
            # 仅对无摘要无关键词的论文检索全文
            if paper.get("abstract") or paper.get("keywords"):
                continue
            text = paper.get(field, "")
            if not text:
                continue
            text_tokens = tokenize(text)
        else:
            text = paper.get(field, "")
            if not text:
                continue
            text_tokens = tokenize(text)

        # 精确匹配（原始查询词）
        exact_matches = query_tokens & text_tokens
        if exact_matches:
            score += len(exact_matches) * weight * 2.0
            matched_fields.append(field)
            matched_terms.update(exact_matches)

        # 扩展匹配（同义词）
        expanded_matches = (expanded_tokens - query_tokens) & text_tokens
        if expanded_matches:
            score += len(expanded_matches) * weight * 0.5
            matched_terms.update(expanded_matches)

    # 加分项
    if paper.get("abstract"):
        if "[兜底提取]" not in paper["abstract"]:
            score *= 1.2
        else:
            score *= 1.05
    if paper.get("keywords"):
        score *= 1.1

    # 匹配字段多样性加分
    if len(matched_fields) >= 3:
        score *= 1.3

    # 查询概念覆盖率加分：优先返回匹配了所有查询概念的论文
    if len(query_tokens) >= 2:
        coverage = len(matched_terms & query_tokens) / len(query_tokens)
        if coverage >= 0.9:
            score *= 2.0  # 覆盖几乎所有查询词
        elif coverage >= 0.7:
            score *= 1.5  # 覆盖大部分查询词
        elif coverage >= 0.5:
            score *= 1.2  # 覆盖半数查询词

    return score, matched_fields, matched_terms


def keyword_search(query, papers, top_n=50, folder_filter=None, exclude_fallback=False):
    """关键词搜索"""
    query_terms = parse_query(query)
    query_tokens = set(t.lower() for t in query_terms)
    query_tokens.update(query_terms)

    expanded_tokens, matched_topics = expand_query(query_terms)

    results = []
    for paper in papers:
        if folder_filter and folder_filter not in paper.get("folder", ""):
            continue
        if paper.get("is_scannable"):
            continue
        fname = paper.get("filename", "")
        if "发明专利" in fname or "专著" in fname:
            continue
        if exclude_fallback and paper.get("abstract", "").startswith("[兜底提取]"):
            continue

        s, matched, terms = score_paper_keyword(paper, query_tokens, expanded_tokens)
        if s > 0:
            results.append((s, matched, terms, paper))

    results.sort(key=lambda x: x[0], reverse=True)

    # 去重（含模糊去重：去掉"论文53-"等编号前缀）
    def _norm_fn(name):
        return re.sub(r'^(?:论文)?\d+[\.\-\s]+', '', name).replace(' ', '')

    seen = set()
    seen_norm = set()
    deduped = []
    for item in results:
        fn = item[3]["filename"]
        norm = _norm_fn(fn)
        if fn in seen or norm in seen_norm:
            continue
        seen.add(fn)
        seen_norm.add(norm)
        deduped.append(item)

    return deduped[:top_n], matched_topics


# ============= 语义搜索 =============

_embeddings_cache = {}

def load_embeddings():
    """加载embedding索引（带缓存）"""
    if 'data' in _embeddings_cache:
        return _embeddings_cache['data']

    if not EMBEDDINGS_PATH.exists():
        return None

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    result = {
        'embeddings': data['embeddings'],
        'filenames': list(data['filenames']),
        'model_name': str(data.get('model_name', 'unknown')),
    }
    # 建立filename→index映射
    result['filename_to_idx'] = {fn: i for i, fn in enumerate(result['filenames'])}
    _embeddings_cache['data'] = result
    return result


_model_cache = {}

def get_embedding_model(model_name):
    """加载embedding模型（带缓存）"""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def semantic_search(query, papers, top_n=50, folder_filter=None):
    """语义搜索：中英文各跑一轮，RRF合并排名"""
    emb_data = load_embeddings()
    if emb_data is None:
        return []

    model = get_embedding_model(emb_data['model_name'])

    # 双语查询
    en_query = _translate_query(query)
    has_en = bool(en_query.strip()) and en_query != query

    queries = [query]
    if has_en:
        queries.append(en_query)

    query_embeddings = model.encode(queries, normalize_embeddings=True)

    # 构建文件名→论文的映射和过滤集
    fn_to_paper = {}
    skip_fns = set()
    for p in papers:
        fn = p.get('filename', '')
        fn_to_paper[fn] = p
        if p.get("is_scannable"):
            skip_fns.add(fn)
        if "发明专利" in fn or "专著" in fn:
            skip_fns.add(fn)
        if folder_filter and folder_filter not in p.get("folder", ""):
            skip_fns.add(fn)

    # 对每个查询生成排名
    all_rankings = []  # list of {filename: rank}
    for qe in query_embeddings:
        sims = emb_data['embeddings'] @ qe
        ranking = {}
        rank = 0
        for idx in np.argsort(sims)[::-1]:
            fn = emb_data['filenames'][idx]
            if fn in skip_fns or fn not in fn_to_paper:
                continue
            if sims[idx] < 0.1:
                break
            rank += 1
            ranking[fn] = rank
        all_rankings.append(ranking)

    # RRF合并排名（k=30）
    k = 30
    rrf_scores = {}
    all_fns = set()
    for ranking in all_rankings:
        all_fns.update(ranking.keys())
    for fn in all_fns:
        score = 0.0
        for ranking in all_rankings:
            if fn in ranking:
                score += 1.0 / (k + ranking[fn])
        rrf_scores[fn] = score

    # 按RRF分数排序，输出 (sim, paper) 格式（sim用中文查询的值供显示）
    sims_cn = emb_data['embeddings'] @ query_embeddings[0]
    fn_to_emb_idx = {fn: i for i, fn in enumerate(emb_data['filenames'])}

    sorted_fns = sorted(rrf_scores.keys(), key=lambda fn: rrf_scores[fn], reverse=True)
    results = []
    for fn in sorted_fns:
        paper = fn_to_paper.get(fn)
        if paper is None:
            continue
        emb_idx = fn_to_emb_idx.get(fn)
        sim = float(sims_cn[emb_idx]) if emb_idx is not None else 0.0
        results.append((sim, paper))
        if len(results) >= top_n:
            break

    return results


# ============= 混合搜索 (RRF) =============

def _is_chinese_query(query):
    """判断查询是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', query))

def hybrid_search(query, papers, top_n=10, folder_filter=None, exclude_fallback=False, extra_queries=None):
    """混合搜索：关键词 + 语义 + 跨语言关键词，使用Reciprocal Rank Fusion (RRF)

    extra_queries: 额外查询列表（如英文翻译），每个查询独立走关键词+语义通道，与主查询RRF融合。
                   这样Claude可以直接传入翻译好的英文查询，无需依赖内置词典。
    """
    # 通道1: 主查询关键词搜索
    kw_results, matched_topics = keyword_search(
        query, papers, top_n=200, folder_filter=folder_filter, exclude_fallback=exclude_fallback
    )

    # 通道2: 主查询语义搜索（已内置CN/EN双通道RRF）
    sem_results = semantic_search(query, papers, top_n=200, folder_filter=folder_filter)

    # 通道3: 跨语言关键词搜索（中文查询→英文词级翻译，搜索英文论文原始字段）
    en_kw_results = []
    if _is_chinese_query(query):
        en_query = _translate_query_wordlevel(query)
        if en_query and en_query.strip():
            en_kw_results, _ = keyword_search(
                en_query, papers, top_n=200, folder_filter=folder_filter, exclude_fallback=exclude_fallback
            )

    # RRF融合
    k = 60  # RRF常数
    paper_scores = defaultdict(float)
    paper_data = {}  # filename → (matched_fields, matched_terms, paper)

    # 通道1: 主查询关键词排名贡献
    for rank, (score, matched, terms, paper) in enumerate(kw_results):
        fn = paper["filename"]
        paper_scores[fn] += 1.0 / (k + rank + 1)
        paper_data[fn] = (matched, terms, paper, score)

    # 通道2: 主查询语义排名贡献
    for rank, (sim, paper) in enumerate(sem_results):
        fn = paper["filename"]
        paper_scores[fn] += 1.0 / (k + rank + 1)
        if fn not in paper_data:
            paper_data[fn] = (["semantic"], set(), paper, 0)

    # 通道3: 内置翻译英文关键词（k=100，作为轻微提升）
    k_en = 100
    for rank, (score, matched, terms, paper) in enumerate(en_kw_results):
        fn = paper["filename"]
        paper_scores[fn] += 1.0 / (k_en + rank + 1)
        if fn not in paper_data:
            paper_data[fn] = (matched, terms, paper, score)

    # 额外查询通道（由调用方提供，如Claude翻译的英文查询）
    extra_sem_results_all = []
    if extra_queries:
        for eq in extra_queries:
            eq = eq.strip()
            if not eq:
                continue
            # 额外查询的关键词通道
            eq_kw, _ = keyword_search(eq, papers, top_n=200, folder_filter=folder_filter,
                                       exclude_fallback=exclude_fallback)
            for rank, (score, matched, terms, paper) in enumerate(eq_kw):
                fn = paper["filename"]
                paper_scores[fn] += 1.0 / (k + rank + 1)
                if fn not in paper_data:
                    paper_data[fn] = (matched, terms, paper, score)

            # 额外查询的语义通道
            eq_sem = semantic_search(eq, papers, top_n=200, folder_filter=folder_filter)
            extra_sem_results_all.extend(eq_sem)
            for rank, (sim, paper) in enumerate(eq_sem):
                fn = paper["filename"]
                paper_scores[fn] += 1.0 / (k + rank + 1)
                if fn not in paper_data:
                    paper_data[fn] = (["semantic"], set(), paper, 0)

    # 按RRF分数排序
    sorted_fns = sorted(paper_scores.keys(), key=lambda fn: paper_scores[fn], reverse=True)

    results = []
    all_sem = sem_results + extra_sem_results_all
    for fn in sorted_fns[:top_n]:
        matched, terms, paper, kw_score = paper_data[fn]
        rrf_score = paper_scores[fn]

        # 获取语义相似度（取所有语义通道的最高值）
        sem_sim = 0.0
        for sim, p in all_sem:
            if p["filename"] == fn:
                sem_sim = max(sem_sim, sim)

        # 在主查询关键词搜索中的排名
        kw_rank = -1
        for i, (_, _, _, p) in enumerate(kw_results):
            if p["filename"] == fn:
                kw_rank = i + 1
                break

        # 在主查询语义搜索中的排名
        sem_rank = -1
        for i, (_, p) in enumerate(sem_results):
            if p["filename"] == fn:
                sem_rank = i + 1
                break

        results.append({
            "paper": paper,
            "rrf_score": rrf_score,
            "kw_score": kw_score,
            "sem_sim": sem_sim,
            "kw_rank": kw_rank,
            "sem_rank": sem_rank,
            "matched_fields": matched,
            "matched_terms": terms,
        })

    return results, matched_topics


# ============= 相似论文推荐 =============

def find_similar(query_name, papers, top_n=10):
    """相似论文推荐"""
    target = None
    for p in papers:
        if query_name.lower() in p["filename"].lower():
            target = p
            break
    if not target:
        return [], [], query_name

    # 构建搜索查询
    search_text = f"{target.get('keywords', '')} {target.get('abstract', '')[:200]}"
    if not search_text.strip():
        search_text = target.get("first_pages_text", "")[:500]

    # 用混合搜索
    results, topics = hybrid_search(search_text, papers, top_n=top_n + 1)

    # 排除自身
    results = [r for r in results if r["paper"]["filename"] != target["filename"]][:top_n]

    return results, topics, target["filename"]


# ============= 格式化输出 =============

def format_results(results, query, matched_topics=None, similar_source=None, search_mode="hybrid"):
    """格式化搜索结果"""
    lines = []

    if similar_source:
        lines.append(f"## 与 \"{similar_source}\" 相似的论文")
    else:
        lines.append(f"## 搜索: \"{query}\" [{search_mode}]")

    if matched_topics:
        lines.append(f"扩展主题: {', '.join(matched_topics)}")

    lines.append(f"找到 {len(results)} 篇相关论文\n")

    for rank, item in enumerate(results, 1):
        if isinstance(item, dict):
            # 混合搜索结果
            p = item["paper"]
            rrf = item["rrf_score"]
            kw_score = item["kw_score"]
            sem_sim = item["sem_sim"]
            kw_rank = item["kw_rank"]
            sem_rank = item["sem_rank"]
            matched = item["matched_fields"]
            terms = item["matched_terms"]

            score_parts = []
            if kw_rank > 0:
                score_parts.append(f"关键词#{kw_rank}")
            if sem_rank > 0:
                score_parts.append(f"语义#{sem_rank}({sem_sim:.2f})")
            score_info = " | ".join(score_parts) if score_parts else ""

            lines.append(f"### [{rank}] RRF: {rrf:.4f}  {score_info}")
        else:
            # 旧格式兼容
            score, matched, terms, p = item
            lines.append(f"### [{rank}] 相关度: {score:.1f}")

        lang = "中" if p["language"] == "zh" else "英"
        year = p["year"] or "?"
        pages = p["page_count"]
        thesis = " 🎓" if p.get("is_thesis") else ""
        source = " 📚Z" if p.get("source") == "zotero" else ""

        lines.append(f"**{p['filename']}** ({lang}, {year}, {pages}页{thesis}{source})")
        lines.append(f"文件夹: {p['folder']}")

        if matched:
            lines.append(f"匹配字段: {', '.join(matched)}")
        if terms:
            display_terms = sorted(terms, key=len, reverse=True)[:10]
            lines.append(f"匹配词: {', '.join(display_terms)}")

        # 显示Zotero元数据
        if p.get("zotero_title"):
            lines.append(f"Zotero标题: {p['zotero_title']}")
        if p.get("zotero_authors"):
            authors = ", ".join(p["zotero_authors"][:3])
            if len(p["zotero_authors"]) > 3:
                authors += f" 等({len(p['zotero_authors'])}人)"
            lines.append(f"作者: {authors}")

        if p["keywords"]:
            lines.append(f"关键词: {p['keywords'][:300]}")
        if p["abstract"]:
            abs_text = p["abstract"][:500]
            if len(p["abstract"]) > 500:
                abs_text += "..."
            lines.append(f"摘要: {abs_text}")

        lines.append(f"路径: {p['path']}")
        lines.append("")

    return "\n".join(lines)


def show_stats():
    """显示索引统计"""
    index = load_index()
    stats = index["stats"]

    print("=== 文献索引统计 v3 ===")
    for k, v in stats.items():
        if k == "top_keywords":
            continue
        if k == "by_method":
            print(f"  提取方法分布:")
            for mk, mv in v.items():
                print(f"    {mk}: {mv}")
        else:
            print(f"  {k}: {v}")

    # 按文件夹统计
    by_folder = defaultdict(lambda: {"total": 0, "with_abs": 0, "thesis": 0, "local": 0, "zotero": 0})
    for p in index["papers"]:
        f = p["folder"]
        by_folder[f]["total"] += 1
        if p["abstract"]:
            by_folder[f]["with_abs"] += 1
        if p.get("is_thesis"):
            by_folder[f]["thesis"] += 1
        if p.get("source") == "zotero":
            by_folder[f]["zotero"] += 1
        else:
            by_folder[f]["local"] += 1

    print("\n=== 按文件夹/分类分布 ===")
    for folder in sorted(by_folder.keys()):
        d = by_folder[folder]
        src = f"本地{d['local']}" + (f"+Z{d['zotero']}" if d['zotero'] else "")
        print(f"  {folder}: {d['total']}篇 ({src}, 摘要{d['with_abs']})")

    # Embedding状态
    if EMBEDDINGS_PATH.exists():
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        print(f"\n=== Embedding索引 ===")
        print(f"  论文数: {len(data['filenames'])}")
        print(f"  向量维度: {data['embeddings'].shape[1]}")
        print(f"  模型: {data.get('model_name', 'unknown')}")
        print(f"  文件大小: {EMBEDDINGS_PATH.stat().st_size/1024/1024:.1f} MB")
    else:
        print(f"\n⚠️ 无Embedding索引 (运行 python3 build_embeddings.py 创建)")

    # 高频关键词
    top_kw = stats.get("top_keywords", [])
    if top_kw:
        print(f"\n=== Top 30 高频关键词 ===")
        for item in top_kw[:30]:
            print(f"  {item['keyword']}: {item['count']}")


def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 为英文论文动态生成cn_topics字段
    for p in data["papers"]:
        if p.get("language") == "en" and not p.get("cn_topics"):
            p["cn_topics"] = _generate_cn_topics(p)
    return data


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    if "--stats" in args:
        show_stats()
        return

    top_n = 10
    folder_filter = None
    topic_mode = False
    year_sort = False
    similar_mode = False
    exclude_fallback = False
    search_mode = "hybrid"  # hybrid, keyword, semantic
    query_parts = []
    also_queries = []  # 额外查询（多查询RRF融合）

    i = 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 2
        elif args[i] == "--folder" and i + 1 < len(args):
            folder_filter = args[i + 1]
            i += 2
        elif args[i] == "--topic":
            topic_mode = True
            i += 1
        elif args[i] == "--year-sort":
            year_sort = True
            i += 1
        elif args[i] == "--similar":
            similar_mode = True
            i += 1
        elif args[i] == "--no-fallback":
            exclude_fallback = True
            i += 1
        elif args[i] == "--keyword":
            search_mode = "keyword"
            i += 1
        elif args[i] == "--semantic":
            search_mode = "semantic"
            i += 1
        elif args[i] == "--hybrid":
            search_mode = "hybrid"
            i += 1
        elif args[i] == "--also" and i + 1 < len(args):
            also_queries.append(args[i + 1])
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    if not query:
        print("请提供搜索关键词")
        return

    # 加载索引
    index = load_index()
    papers = index["papers"]

    if similar_mode:
        results, topics, source = find_similar(query, papers, top_n=top_n)
        output = format_results(results, query, matched_topics=topics, similar_source=source)
        print(output)
        return

    # 根据模式搜索
    if search_mode == "keyword":
        results, topics = keyword_search(query, papers, top_n=top_n,
                                         folder_filter=folder_filter,
                                         exclude_fallback=exclude_fallback)
        # 转换为统一格式
        formatted = []
        for score, matched, terms, paper in results:
            formatted.append({
                "paper": paper,
                "rrf_score": score,
                "kw_score": score,
                "sem_sim": 0,
                "kw_rank": 0,
                "sem_rank": 0,
                "matched_fields": matched,
                "matched_terms": terms,
            })
        output = format_results(formatted, query, matched_topics=topics, search_mode="keyword")

    elif search_mode == "semantic":
        if not EMBEDDINGS_PATH.exists():
            print("⚠️ 无Embedding索引，回退到关键词搜索")
            print("  请先运行: python3 build_embeddings.py")
            search_mode = "keyword"
            results, topics = keyword_search(query, papers, top_n=top_n,
                                             folder_filter=folder_filter)
            formatted = []
            for score, matched, terms, paper in results:
                formatted.append({
                    "paper": paper,
                    "rrf_score": score,
                    "kw_score": score,
                    "sem_sim": 0,
                    "kw_rank": 0,
                    "sem_rank": 0,
                    "matched_fields": matched,
                    "matched_terms": terms,
                })
            output = format_results(formatted, query, matched_topics=topics, search_mode="keyword(fallback)")
        else:
            sem_results = semantic_search(query, papers, top_n=top_n, folder_filter=folder_filter)
            formatted = []
            for rank, (sim, paper) in enumerate(sem_results):
                formatted.append({
                    "paper": paper,
                    "rrf_score": sim,
                    "kw_score": 0,
                    "sem_sim": sim,
                    "kw_rank": 0,
                    "sem_rank": rank + 1,
                    "matched_fields": ["semantic"],
                    "matched_terms": set(),
                })
            output = format_results(formatted, query, search_mode="semantic")

    else:  # hybrid
        if not EMBEDDINGS_PATH.exists():
            # 无embedding，回退到纯关键词搜索
            results, topics = keyword_search(query, papers, top_n=top_n,
                                             folder_filter=folder_filter,
                                             exclude_fallback=exclude_fallback)
            formatted = []
            for score, matched, terms, paper in results:
                formatted.append({
                    "paper": paper,
                    "rrf_score": score,
                    "kw_score": score,
                    "sem_sim": 0,
                    "kw_rank": 0,
                    "sem_rank": 0,
                    "matched_fields": matched,
                    "matched_terms": terms,
                })
            output = format_results(formatted, query, matched_topics=topics, search_mode="keyword(no embedding)")
        else:
            results, topics = hybrid_search(query, papers, top_n=top_n,
                                            folder_filter=folder_filter,
                                            exclude_fallback=exclude_fallback,
                                            extra_queries=also_queries if also_queries else None)
            mode_label = f"hybrid+{len(also_queries)}q" if also_queries else "hybrid"
            output = format_results(results, query, matched_topics=topics, search_mode=mode_label)

    if year_sort and search_mode != "semantic":
        # 年份排序模式下重排
        print("(结果已按相关度排序，添加--year-sort仅在keyword模式下按年份排序)")

    print(output)


if __name__ == "__main__":
    main()
