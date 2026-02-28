#!/usr/bin/env python3
"""
文献索引构建器 v3 - Zotero集成 + jieba分词
改进点(相比v2):
  1. 扫描Zotero存储目录（跨文件系统自动处理）
  2. 从Zotero sqlite读取元数据（分类/标签/标题/作者）
  3. 使用jieba分词替代n-gram（大幅提升搜索准确性）
  4. 按文件名去重（同一论文在本地和Zotero都有时只保留本地版）

运行方式:
  python3 build_paper_index.py            # 全量重建
  python3 build_paper_index.py --incr     # 增量更新（只处理新文件）
  python3 build_paper_index.py --no-zotero  # 不扫描Zotero
"""

import fitz  # PyMuPDF
import json
import os
import re
import sys
import time
import sqlite3
import jieba
from pathlib import Path
from collections import Counter, defaultdict

# 关闭MuPDF冗余警告，避免个别异常注释PDF刷屏拖慢构建
try:
    fitz.TOOLS.mupdf_display_warnings(False)
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

# 跳过明显的非论文附件（补充材料/审稿文件/翻译件）
SKIP_FILENAME_PATTERNS = [
    r'(?i)\b(supplement|supplementary|supporting information|transparent peer review|peer review file|supplementary data|supplementary figs?)\b',
    r'(?i)\bsupplemental material\b',
    r'中文翻译',
    r'全文中文翻译',
    r'中译全文',
    r'中文全译',
    r'补充材料',
    r'图像图题与表题中文翻译',
]


def should_skip_pdf(filename):
    name = filename.strip()
    for pat in SKIP_FILENAME_PATTERNS:
        if re.search(pat, name):
            return True
    return False


def _is_control_garbled(s):
    return bool(re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', s or ""))


def is_title_low_quality(title):
    """判断标题是否低质量（乱码/模板词/学位论文泛称）"""
    t = (title or "").strip().lower()
    if not t:
        return True
    if _is_control_garbled(t) or '�' in t:
        return True
    generic = [
        "article", "research article", "research papers", "original research article",
        "硕士学位论文", "博士学位论文", "学位论文"
    ]
    if t in generic:
        return True
    if any(k in t for k in ["copyright", "unauthenticated", "downloaded"]):
        return True
    return False

# ============= 路径配置（从 config.py 读取，不存在时用默认值）=============
try:
    from config import PDF_DIR as BASE_DIR, ZOTERO_DIR, INDEX_PATH as OUTPUT_JSON
    ZOTERO_STORAGE = Path(ZOTERO_DIR).expanduser() / "storage" if ZOTERO_DIR and str(ZOTERO_DIR) else Path("")
    ZOTERO_DB = Path(ZOTERO_DIR).expanduser() / "zotero.sqlite" if ZOTERO_DIR and str(ZOTERO_DIR) else Path("")
    BASE_DIR = Path(BASE_DIR).expanduser()
except ImportError:
    # 未配置时的默认值（请修改 config.py）
    BASE_DIR = Path("~/论文").expanduser()
    ZOTERO_STORAGE = Path("~/Zotero/storage").expanduser()
    ZOTERO_DB = Path("~/Zotero/zotero.sqlite").expanduser()
    OUTPUT_JSON = Path(__file__).parent / "paper_index.json"
OUTPUT_MD = OUTPUT_JSON.parent / "paper_index_readable.md"

# ============= jieba 领域词典 =============
DOMAIN_WORDS = [
    "物候", "物候期", "物候变化", "返青期", "枯黄期", "生长季",
    "蒸散", "蒸散发", "蒸腾", "潜在蒸散", "实际蒸散",
    "径流", "产流", "基流", "枯水", "洪水", "洪峰",
    "径流量", "径流深", "径流系数", "天然径流",
    "植被覆盖", "植被恢复", "植被指数", "植被动态",
    "遥感反演", "遥感监测", "遥感数据",
    "水源涵养", "水源涵养量", "林冠截留",
    "生态流量", "生态需水", "生态基流", "环境流量",
    "黄土高原", "西辽河", "海河流域", "黄河流域", "长江流域",
    "气候变化", "气候变暖", "全球变暖", "极端气候",
    "退耕还林", "退耕还草", "水土保持",
    "淤地坝", "鱼鳞坑", "水平阶", "梯田",
    "碳循环", "碳汇", "碳储量", "净初级生产力",
    "归因分析", "弹性系数", "敏感性分析",
    "结构方程", "通径分析", "因果分析",
    "随机森林", "机器学习", "深度学习",
    "生态水文", "水文模型", "水文效应",
    "干旱指数", "干旱事件", "干旱胁迫",
    "时空变化", "时空格局", "时空分布",
    "水文气候", "水热耦合", "水量平衡",
]
for w in DOMAIN_WORDS:
    jieba.add_word(w)

# 预加载jieba词典（避免首次调用延迟）
jieba.initialize()

# ============= 停用词 =============
STOPWORDS_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'it',
    'its', 'this', 'that', 'these', 'those', 'we', 'our', 'they', 'their',
    'he', 'she', 'his', 'her', 'not', 'no', 'than', 'more', 'also',
    'between', 'during', 'under', 'over', 'about', 'into', 'through',
    'using', 'used', 'based', 'both', 'each', 'such', 'which', 'where',
    'when', 'how', 'what', 'who', 'whom', 'there', 'here', 'all', 'any',
    'some', 'most', 'very', 'well', 'while', 'however', 'although',
    'because', 'since', 'after', 'before', 'then', 'only', 'just',
    'other', 'new', 'first', 'two', 'three', 'one', 'many', 'much',
    'high', 'low', 'large', 'small', 'long', 'short', 'different',
    'same', 'main', 'major', 'important', 'significant', 'total', 'per',
    'respectively', 'including', 'particularly', 'especially', 'overall',
    'among', 'within', 'without', 'above', 'below', 'across', 'along',
    'further', 'still', 'even', 'thus', 'therefore', 'hence', 'moreover',
    'results', 'study', 'studies', 'research', 'paper', 'analysis',
    'method', 'methods', 'data', 'found', 'showed', 'show', 'shows',
    'indicate', 'indicates', 'indicated', 'suggest', 'suggests',
    'observed', 'compared', 'effect', 'effects', 'impact', 'impacts',
    'increase', 'increased', 'decrease', 'decreased', 'change', 'changes',
    'changed', 'significantly', 'higher', 'lower',
    'university', 'institute', 'department', 'college', 'school',
    'laboratory', 'center', 'journal', 'proceedings', 'press',
    'beijing', 'china', 'usa', 'doi', 'http', 'https', 'www',
    'fig', 'figure', 'table', 'section', 'abstract', 'keywords',
    'acknowledgement', 'acknowledgements', 'references', 'appendix',
}
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

# OCR 支持（可选）
try:
    import pytesseract
    from PIL import Image
    import io
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


# ============= v3: jieba分词 =============
def tokenize(text):
    """分词（中英文混合）- v3使用jieba切词，替代n-gram"""
    tokens = set()

    # 英文单词
    en_words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{1,}', text)
    tokens.update(w.lower() for w in en_words if len(w) >= 2)

    # 中文：jieba搜索模式分词（兼顾长短词）
    zh_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', text))
    if zh_text:
        words = jieba.cut_for_search(zh_text)
        tokens.update(w for w in words if len(w) >= 2 and w not in STOPWORDS_ZH)

    return sorted(tokens)


# ============= v3: Zotero元数据 =============
def load_zotero_metadata(db_path):
    """从Zotero数据库读取每篇PDF的元数据
    返回 {filename: {title, authors, collections, tags, year}}
    """
    if not db_path.exists():
        print(f"  Zotero数据库不存在: {db_path}")
        return {}

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    except Exception:
        # 如果只读模式失败，复制到临时文件
        import shutil, tempfile
        tmp = Path(tempfile.mktemp(suffix='.sqlite'))
        shutil.copy2(db_path, tmp)
        conn = sqlite3.connect(str(tmp))

    cur = conn.cursor()
    metadata = {}

    # 获取所有PDF附件及其父条目ID
    cur.execute('''
        SELECT ia.path, ia.parentItemID
        FROM itemAttachments ia
        WHERE ia.contentType = 'application/pdf' AND ia.path IS NOT NULL
    ''')
    pdf_items = cur.fetchall()

    # 获取字段ID映射
    cur.execute("SELECT fieldID, fieldName FROM fields")
    field_map = {name: fid for fid, name in cur.fetchall()}
    title_fid = field_map.get('title')
    date_fid = field_map.get('date')

    for path, parent_id in pdf_items:
        if not path or not path.startswith('storage:'):
            continue
        filename = path[len('storage:'):]
        if not filename.endswith('.pdf'):
            continue

        meta = {'title': '', 'authors': [], 'collections': [], 'tags': [], 'year': ''}

        if parent_id:
            # 标题
            if title_fid:
                cur.execute('''
                    SELECT idv.value FROM itemData id
                    JOIN itemDataValues idv ON id.valueID = idv.valueID
                    WHERE id.itemID = ? AND id.fieldID = ?
                ''', (parent_id, title_fid))
                row = cur.fetchone()
                if row:
                    meta['title'] = row[0]

            # 年份
            if date_fid:
                cur.execute('''
                    SELECT idv.value FROM itemData id
                    JOIN itemDataValues idv ON id.valueID = idv.valueID
                    WHERE id.itemID = ? AND id.fieldID = ?
                ''', (parent_id, date_fid))
                row = cur.fetchone()
                if row:
                    m = re.search(r'(19|20)\d{2}', str(row[0]))
                    if m:
                        meta['year'] = m.group(0)

            # 作者
            cur.execute('''
                SELECT c.firstName, c.lastName FROM itemCreators ic
                JOIN creators c ON ic.creatorID = c.creatorID
                WHERE ic.itemID = ? ORDER BY ic.orderIndex
            ''', (parent_id,))
            meta['authors'] = [f"{r[1]} {r[0]}".strip() for r in cur.fetchall()]

            # 分类（collections）
            cur.execute('''
                SELECT c.collectionName FROM collectionItems ci
                JOIN collections c ON ci.collectionID = c.collectionID
                WHERE ci.itemID = ?
            ''', (parent_id,))
            meta['collections'] = [r[0] for r in cur.fetchall()]

            # 标签
            cur.execute('''
                SELECT t.name FROM itemTags it
                JOIN tags t ON it.tagID = t.tagID
                WHERE it.itemID = ?
            ''', (parent_id,))
            meta['tags'] = [r[0] for r in cur.fetchall()]

        metadata[filename] = meta

    conn.close()
    return metadata


# ============= PDF提取函数（保留v2全部逻辑） =============

def auto_generate_keywords(abstract, first_pages_text, lang):
    """从摘要和首页文本自动生成关键词"""
    clean_abstract = re.sub(r'^\[兜底提取\]\s*', '', abstract or "")
    text = clean_abstract + " " + (first_pages_text or "")[:1000]
    if len(text.strip()) < 50:
        return ""

    if lang == "zh":
        segments = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        freq = Counter(segments)
        keywords = []
        seen = set()
        for word, cnt in freq.most_common(50):
            if word in STOPWORDS_ZH or len(word) < 2:
                continue
            if any(word in kw for kw in seen if len(kw) > len(word)):
                continue
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= 8:
                break
        return "[自动] " + "; ".join(keywords) if keywords else ""
    else:
        words = re.findall(r'[a-zA-Z][a-zA-Z-]{2,}', text)
        freq = Counter(w.lower() for w in words)
        keywords = []
        for word, cnt in freq.most_common(60):
            if word in STOPWORDS_EN or len(word) < 3 or cnt < 2:
                continue
            keywords.append(word)
            if len(keywords) >= 8:
                break
        bigrams = []
        words_list = [w.lower() for w in words if w.lower() not in STOPWORDS_EN and len(w) > 2]
        for i in range(len(words_list) - 1):
            bg = f"{words_list[i]} {words_list[i+1]}"
            bigrams.append(bg)
        bg_freq = Counter(bigrams)
        top_bigrams = [bg for bg, cnt in bg_freq.most_common(5) if cnt >= 2]
        combined = top_bigrams[:4] + [kw for kw in keywords if not any(kw in bg for bg in top_bigrams)]
        return "[auto] " + "; ".join(combined[:8]) if combined else ""


def detect_language(text):
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total = len(text.strip())
    if total == 0:
        return "unknown"
    return "zh" if chinese_chars / total > 0.15 else "en"


def is_thesis(text, page_count):
    if page_count < 30:
        return False
    thesis_markers = [
        '学位论文', '硕士', '博士', '学位类别', '指导教师', '导师',
        '论文答辩', '学科专业', '研究方向', '论文提交',
        'Thesis', 'Dissertation', 'degree', 'supervisor',
        'Submitted to', 'Fulfillment', 'Requirements for',
    ]
    text_lower = text.lower()
    matches = sum(1 for m in thesis_markers if m.lower() in text_lower)
    return matches >= 2


def extract_abstract_zh(text):
    patterns = [
        r'摘\s*要[：:\s]*(.+?)(?:关\s*键\s*词|关键字|Key\s*words|Keywords)',
        r'中文摘要[：:\s\d]*(.+?)(?:关\s*键\s*词|关键字|Key\s*words|Keywords|英文摘要|Abstract|ABSTRACT)',
        r'内容摘要[：:\s]*(.+?)(?:关\s*键\s*词|关键字)',
        r'内容提要[：:\s]*(.+?)(?:关\s*键\s*词|关键字)',
        r'摘要[（(]\s*Abstract\s*[)）][：:\s]*(.+?)(?:关\s*键\s*词|关键字|Key\s*words)',
        r'摘\s*要[：:\s]*(.+?)(?:ABSTRACT|Abstract\s)',
        r'摘\s*要\s*[IⅠ]?\s*\n(.+?)(?:关\s*键\s*词|关键字|Key\s*words)',
        r'摘\s*要[：:\s]*(.{200,}?)(?:\n\s*(?:目\s*录|第\s*[一二三1-3]\s*章|1\s+引言|1\s+绪论|前\s*言))',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            abstract = m.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]
    return ""


def extract_abstract_en(text):
    terminators = (
        r'(?:'
        r'K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S'
        r'|Keywords?|Key\s*words|INDEX\s+TERMS|Index\s+terms'
        r'|Introduction|1\s*\.?\s*Introduction|I\.\s+Introduction'
        r'|Background\s*(?:&|and)\s*Summary'
        r'|Results?\s*(?:&|and)\s*Discussion'
        r'|Main\s*\n'
        r'|\n\s*\n\s*(?:\d+\.?\s+\w)'
        r'|©|Copyright'
        r')'
    )
    patterns = [
        rf'Abstract[.:\s—–-]*(.+?)(?:{terminators})',
        rf'A\s*B\s*S\s*T\s*R\s*A\s*C\s*T[:\s]*(.+?)(?:{terminators})',
        rf'SUMMARY[:\s]*(.+?)(?:{terminators})',
        rf'(?:Highlights?.+?)Abstract[.:\s]*(.+?)(?:{terminators})',
        r'Abstract[.:\s—–-]*(.{200,2000?}?)(?:\n\s*\n)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            abstract = m.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]
    return ""


def extract_fallback_abstract(text, lang):
    skip_markers = [
        'http', 'doi:', 'issn', '©', 'copyright', 'received',
        '收稿日期', '基金项目', '作者简介', '目录', '参考文献',
        '发明专利', '权利要求', 'cite this', 'accepted',
        'published online', 'supplementary', 'supplement of',
    ]

    def is_content_paragraph(clean, lang, strict_skip=True):
        if len(clean) < 150:
            return False
        check_text = clean.lower() if strict_skip else clean[:100].lower()
        if any(m in check_text for m in skip_markers):
            return False
        if lang == "zh":
            return len(re.findall(r'[\u4e00-\u9fff]', clean)) > 50
        else:
            words = clean.split()
            return len(words) > 40 and any(c == '.' for c in clean)

    lines = text.split('\n')
    affiliation_prefix_re = re.compile(r'^[a-z]{1,2}[A-Z]')
    non_content_re = re.compile(
        r'(?:University|Institute|Department|College|Academy|Laboratory)'
        r'|(?:Edited by|Contributed by|Communicated by|Significance)'
        r'|(?:@|\.edu|\.ac\.|\.org)'
        r'|(?:\d{5})'
        r'|(?:Received |Accepted |Available online|Published online)',
        re.IGNORECASE
    )
    found_start = None
    for i, line in enumerate(lines[:50]):
        stripped = line.strip()
        if len(stripped) < 40:
            continue
        if non_content_re.search(stripped) or affiliation_prefix_re.match(stripped):
            continue
        if re.match(r'^[A-Z][a-z]', stripped) and '.' not in stripped[:5]:
            found_start = i
            break
    if found_start is not None:
        candidate = []
        for j in range(found_start, min(found_start + 25, len(lines))):
            l = lines[j].strip()
            if re.match(r'^(?:Introduction|Background\b|Results|Methods|Main$|1\s*[\.\s])', l, re.IGNORECASE):
                break
            if re.match(r'^(?:Background\s*(?:&|and)\s*Summary)', l, re.IGNORECASE):
                break
            if re.match(r'^(?:water cycle|Keywords?|K\s*E\s*Y)', l, re.IGNORECASE) and len(l) < 200:
                break
            candidate.append(l)
        joined = re.sub(r'\s+', ' ', ' '.join(candidate)).strip()
        if is_content_paragraph(joined, lang, strict_skip=False):
            return "[兜底提取] " + joined[:2000]

    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        clean = re.sub(r'\s+', ' ', para).strip()
        if is_content_paragraph(clean, lang):
            return "[兜底提取] " + clean[:2000]

    current_block = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                joined = re.sub(r'\s+', ' ', ' '.join(current_block)).strip()
                if is_content_paragraph(joined, lang):
                    return "[兜底提取] " + joined[:2000]
                current_block = []
        else:
            current_block.append(stripped)
    if current_block:
        joined = re.sub(r'\s+', ' ', ' '.join(current_block)).strip()
        if is_content_paragraph(joined, lang):
            return "[兜底提取] " + joined[:2000]

    full_text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.。])\s+', full_text)
    consecutive = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:
            if consecutive:
                break
            continue
        if any(m in sent.lower() for m in skip_markers):
            if consecutive:
                break
            continue
        if re.match(r'^[a-z](?:Department|University|Institute|School)', sent):
            continue
        consecutive.append(sent)
        if len(consecutive) >= 3:
            joined = '. '.join(consecutive)
            if len(joined) > 200:
                return "[兜底提取] " + joined[:2000]
    return ""


def extract_keywords_zh(text):
    patterns = [
        r'关\s*键\s*词[：:\s]*(.+?)(?:\n|Abstract|ABSTRACT|中图分类号|分类号|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            kw = m.group(1).strip()
            kw = re.sub(r'\s+', ' ', kw)
            return kw[:500]
    return ""


def extract_keywords_en(text):
    patterns = [
        r'Key\s*words?[：:\s]*(.+?)(?:\n\n|\n\s*\d|Introduction|Copyright|©|$)',
        r'KEY\s*WORDS?[：:\s]*(.+?)(?:\n\n|\n\s*\d|Introduction|$)',
        r'K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S\s*\n(.+?)(?:\n\s*\n|\n\s*\d|\n\s*(?:INTRODUCTION|Introduction))',
        r'Index\s+terms?[：:\s]*(.+?)(?:\n\n|\n\s*\d|Introduction|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            kw = m.group(1).strip()
            kw = re.sub(r'\s+', ' ', kw)
            if len(kw) > 3:
                return kw[:500]
    return ""


def extract_keywords(text, lang):
    kw_zh = extract_keywords_zh(text)
    kw_en = extract_keywords_en(text)
    if kw_zh and kw_en:
        return f"{kw_zh} | {kw_en}"
    return kw_zh or kw_en


def extract_year(filename, text):
    m = re.search(r'(19|20)\d{2}', filename)
    if m:
        return m.group(0)
    years = re.findall(r'(20[0-2]\d|19\d{2})', text[:3000])
    if years:
        common = Counter(years).most_common(3)
        for y, _ in common:
            if 1990 <= int(y) <= 2026:
                return y
    return ""


def extract_title_from_text(text, lang):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
    skip_words = [
        'doi:', 'http', 'journal', 'volume', 'issn', '收稿日期',
        '基金项目', '作者简介', '分类号', '密级', '编号',
        'scientific data', 'scientific reports', 'www.nature.com',
    ]
    for line in lines[:8]:
        if any(skip in line.lower() for skip in skip_words):
            continue
        if len(line) > 5 and len(line) < 200:
            return line[:200]
    return ""


def ocr_scanned_pdf(doc, max_pages=3):
    if not HAS_OCR:
        return ""
    text_parts = []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            langs = 'chi_sim+eng' if os.path.exists('/usr/share/tesseract-ocr/5/tessdata/chi_sim.traineddata') else 'eng'
            page_text = pytesseract.image_to_string(img, lang=langs)
            text_parts.append(page_text)
        except Exception:
            pass
    return "\n".join(text_parts)


# ============= 核心处理函数 =============

def process_pdf(pdf_path, source="local", zotero_meta=None):
    """处理单个PDF，提取元数据
    source: "local" 或 "zotero"
    zotero_meta: Zotero元数据字典（仅zotero来源）
    """
    # 确定folder字段
    if source == "local":
        try:
            folder = str(pdf_path.parent.relative_to(BASE_DIR))
        except ValueError:
            folder = pdf_path.parent.name
    elif source == "zotero" and zotero_meta:
        if zotero_meta.get("collections"):
            folder = " | ".join(zotero_meta["collections"])
        else:
            folder = "Zotero/未分类"
    else:
        folder = "unknown"

    result = {
        "path": str(pdf_path),
        "folder": folder,
        "filename": pdf_path.name,
        "source": source,
        "language": "unknown",
        "title_extracted": "",
        "abstract": "",
        "keywords": "",
        "year": "",
        "first_pages_text": "",
        "text_length": 0,
        "page_count": 0,
        "is_scannable": False,
        "is_thesis": False,
        "extraction_method": "standard",
    }

    # v3: 写入Zotero元数据
    if zotero_meta:
        result["zotero_title"] = zotero_meta.get("title", "")
        result["zotero_authors"] = zotero_meta.get("authors", [])
        result["zotero_collections"] = zotero_meta.get("collections", [])
        result["zotero_tags"] = zotero_meta.get("tags", [])
        # Zotero年份优先
        if zotero_meta.get("year"):
            result["year"] = zotero_meta["year"]

    try:
        doc = fitz.open(str(pdf_path))
        result["page_count"] = len(doc)

        text_3p = ""
        for i in range(min(3, len(doc))):
            text_3p += doc[i].get_text() + "\n"

        if len(text_3p.strip()) < 100:
            result["is_scannable"] = True
            ocr_text = ocr_scanned_pdf(doc, max_pages=3)
            if len(ocr_text.strip()) > 100:
                text_3p = ocr_text
                result["is_scannable"] = False
                result["extraction_method"] = "ocr"
            else:
                doc.close()
                return result

        thesis = is_thesis(text_3p, result["page_count"])
        result["is_thesis"] = thesis

        if thesis:
            text = ""
            for i in range(min(10, len(doc))):
                text += doc[i].get_text() + "\n"
            result["extraction_method"] = "extended"
        else:
            text = text_3p

        result["text_length"] = len(text.strip())

        lang = detect_language(text)
        result["language"] = lang

        if lang == "zh":
            abstract = extract_abstract_zh(text)
        else:
            abstract = extract_abstract_en(text)

        if not abstract:
            if lang == "zh":
                abstract = extract_abstract_en(text)
            else:
                abstract = extract_abstract_zh(text)

        if not abstract:
            abstract = extract_fallback_abstract(text, lang)
            if abstract:
                result["extraction_method"] = "fallback"

        result["abstract"] = abstract
        result["keywords"] = extract_keywords(text, lang)

        if not result["year"]:
            result["year"] = extract_year(pdf_path.name, text)

        result["title_extracted"] = extract_title_from_text(text, lang)

        clean_text = re.sub(r'\s+', ' ', text)
        max_text = 6000 if thesis else 3000
        result["first_pages_text"] = clean_text[:max_text]

        doc.close()

        # 若提取标题质量较差，优先使用Zotero标题覆盖
        if result.get("zotero_title") and is_title_low_quality(result.get("title_extracted", "")):
            result["title_extracted"] = result["zotero_title"][:200]

        # 摘要兜底补全：优先首段正文，其次Zotero标题/文件名
        if not result["abstract"]:
            if result.get("first_pages_text"):
                snippet = result["first_pages_text"][:600].strip()
                if snippet:
                    result["abstract"] = f"[兜底补全] {snippet}"
            if not result["abstract"] and result.get("zotero_title"):
                result["abstract"] = f"[兜底补全] {result['zotero_title']}"
            if not result["abstract"]:
                result["abstract"] = f"[兜底补全] {result['filename']}"
            if result["abstract"] and result.get("extraction_method") == "standard":
                result["extraction_method"] = "fallback"

        # 关键词兜底补全
        if not result["keywords"]:
            result["keywords"] = auto_generate_keywords(
                result.get("abstract", ""), result.get("first_pages_text", ""), lang
            )
        if not result["keywords"] and result.get("zotero_title"):
            toks = tokenize(result["zotero_title"])[:8]
            if toks:
                result["keywords"] = "[自动] " + "; ".join(toks)
        if not result["keywords"]:
            fallback_title = result.get("zotero_title") or result.get("title_extracted") or result.get("filename", "")
            toks = tokenize(fallback_title)[:8]
            if toks:
                result["keywords"] = "[自动] " + "; ".join(toks)
        if not result["keywords"]:
            fallback_text = f"{result.get('title_extracted', '')} {result.get('filename', '')}"
            zh_chunks = re.findall(r'[\u4e00-\u9fff]{2,8}', fallback_text)
            if zh_chunks:
                result["keywords"] = "[自动] " + "; ".join(list(dict.fromkeys(zh_chunks))[:8])

        # v3: 预计算tokens（使用jieba分词）
        result["tokens"] = {}
        for field in ("filename", "keywords", "abstract", "title_extracted", "folder"):
            val = result.get(field, "")
            if val:
                result["tokens"][field] = tokenize(val)
        # v3: 额外索引Zotero元数据
        zotero_text_parts = []
        if result.get("zotero_title"):
            zotero_text_parts.append(result["zotero_title"])
        if result.get("zotero_tags"):
            zotero_text_parts.append(" ".join(result["zotero_tags"]))
        if result.get("zotero_collections"):
            zotero_text_parts.append(" ".join(result["zotero_collections"]))
        if zotero_text_parts:
            result["tokens"]["zotero_meta"] = tokenize(" ".join(zotero_text_parts))

    except Exception as e:
        result["error"] = str(e)

    return result


# ============= 索引构建 =============

def build_index(incremental=False, include_zotero=True):
    """构建完整索引"""
    print(f"=== 文献索引构建器 v3 ===")

    # 1. 扫描本地目录
    print(f"\n[1/4] 扫描本地目录: {BASE_DIR}")
    local_all = sorted(BASE_DIR.rglob("*.pdf"))
    local_pdfs = [p for p in local_all if not should_skip_pdf(p.name)]
    local_skipped = len(local_all) - len(local_pdfs)
    print(f"  本地PDF: {len(local_pdfs)}篇" + (f" (跳过附件{local_skipped}篇)" if local_skipped else ""))

    # 2. 扫描Zotero目录
    zotero_pdfs = []
    zotero_meta_map = {}
    if include_zotero and ZOTERO_STORAGE.exists():
        print(f"\n[2/4] 扫描Zotero目录: {ZOTERO_STORAGE}")
        zotero_all = sorted(ZOTERO_STORAGE.rglob("*.pdf"))
        zotero_pdfs = [p for p in zotero_all if not should_skip_pdf(p.name)]
        zotero_skipped = len(zotero_all) - len(zotero_pdfs)
        print(f"  Zotero PDF: {len(zotero_pdfs)}篇" + (f" (跳过附件{zotero_skipped}篇)" if zotero_skipped else ""))

        # 读取Zotero元数据
        print(f"  读取Zotero数据库元数据...")
        zotero_meta_map = load_zotero_metadata(ZOTERO_DB)
        print(f"  获取到 {len(zotero_meta_map)} 篇元数据")
    else:
        print(f"\n[2/4] 跳过Zotero扫描")

    # 3. 去重：本地优先
    local_filenames = {p.name for p in local_pdfs}
    zotero_unique = [p for p in zotero_pdfs if p.name not in local_filenames]
    duplicates = len(zotero_pdfs) - len(zotero_unique)
    all_pdfs_info = [(p, "local", None) for p in local_pdfs]
    for p in zotero_unique:
        meta = zotero_meta_map.get(p.name, None)
        all_pdfs_info.append((p, "zotero", meta))

    # 同时为本地PDF补充Zotero元数据（如果有）
    for i, (p, source, meta) in enumerate(all_pdfs_info):
        if source == "local" and p.name in zotero_meta_map:
            all_pdfs_info[i] = (p, "local", zotero_meta_map[p.name])

    total = len(all_pdfs_info)
    print(f"\n[3/4] 去重后总计: {total}篇 (本地{len(local_pdfs)} + Zotero独有{len(zotero_unique)}, 重复跳过{duplicates})")

    # 增量模式处理
    existing_papers = []
    existing_filenames = set()
    if incremental and OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            old_index = json.load(f)
            existing_papers = old_index.get("papers", [])
            existing_filenames = {p["filename"] for p in existing_papers}

        # 移除不再存在的文件
        current_filenames = {info[0].name for info in all_pdfs_info}
        removed = existing_filenames - current_filenames
        if removed:
            existing_papers = [p for p in existing_papers if p["filename"] not in removed]
            print(f"  移除 {len(removed)} 个已删除文件的索引")

        new_pdfs_info = [info for info in all_pdfs_info if info[0].name not in existing_filenames]
        print(f"  增量模式: 跳过 {total - len(new_pdfs_info)} 个已索引, 处理 {len(new_pdfs_info)} 个新文件")
        all_pdfs_info = new_pdfs_info
    else:
        existing_papers = []

    # 4. 处理所有PDF
    print(f"\n[4/4] 处理PDF...")
    papers = list(existing_papers)
    errors = []
    scanned = []
    start_time = time.time()

    for i, (pdf_path, source, meta) in enumerate(all_pdfs_info):
        if (i + 1) % 50 == 0 or (i + 1) == len(all_pdfs_info):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_pdfs_info) - i - 1) / speed if speed > 0 else 0
            print(f"  进度: {i+1}/{len(all_pdfs_info)} ({elapsed:.0f}s, {speed:.1f}篇/s, ETA {eta:.0f}s)")

        result = process_pdf(pdf_path, source=source, zotero_meta=meta)
        papers.append(result)

        if result.get("error"):
            errors.append(result["filename"])
        if result.get("is_scannable"):
            scanned.append(result["filename"])

    elapsed = time.time() - start_time

    # 最终去重（按文件名，保留第一个=本地优先）
    # v3.1: 模糊去重 - 去掉"论文53-"、"39."等编号前缀后比较
    def normalize_filename(name):
        """标准化文件名用于去重比较"""
        # 去掉 '论文53-' '39.' '11.' 等前缀编号
        name = re.sub(r'^(?:论文)?\d+[\.\-\s]+', '', name)
        # 去掉多余空格
        name = name.replace(' ', '')
        return name

    seen_filenames = set()     # 原始文件名去重
    seen_normalized = set()    # 标准化文件名去重
    deduped = []
    fuzzy_dupes = 0
    for p in papers:
        if p["filename"] in seen_filenames:
            continue
        norm = normalize_filename(p["filename"])
        if norm in seen_normalized:
            fuzzy_dupes += 1
            continue
        seen_filenames.add(p["filename"])
        seen_normalized.add(norm)
        deduped.append(p)
    papers = deduped
    if fuzzy_dupes:
        print(f"  模糊去重: 移除 {fuzzy_dupes} 篇编号前缀重复论文")

    # 统计
    stats = {
        "total_papers": len(papers),
        "local_papers": sum(1 for p in papers if p.get("source") == "local"),
        "zotero_papers": sum(1 for p in papers if p.get("source") == "zotero"),
        "with_abstract": sum(1 for p in papers if p["abstract"]),
        "with_keywords": sum(1 for p in papers if p["keywords"]),
        "chinese_papers": sum(1 for p in papers if p["language"] == "zh"),
        "english_papers": sum(1 for p in papers if p["language"] == "en"),
        "thesis_papers": sum(1 for p in papers if p.get("is_thesis")),
        "scanned_papers": sum(1 for p in papers if p.get("is_scannable")),
        "error_papers": sum(1 for p in papers if p.get("error")),
        "with_zotero_meta": sum(1 for p in papers if p.get("zotero_title")),
        "by_method": {
            "standard": sum(1 for p in papers if p.get("extraction_method") == "standard"),
            "extended": sum(1 for p in papers if p.get("extraction_method") == "extended"),
            "fallback": sum(1 for p in papers if p.get("extraction_method") == "fallback"),
            "ocr": sum(1 for p in papers if p.get("extraction_method") == "ocr"),
        },
        "abstract_with_fallback": sum(1 for p in papers if p["abstract"] and "[兜底提取]" in p["abstract"]),
        "keywords_auto": sum(1 for p in papers if p["keywords"] and (p["keywords"].startswith("[自动]") or p["keywords"].startswith("[auto]"))),
        "build_time_seconds": round(elapsed, 1),
        "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v3",
    }

    # 高频关键词
    all_keywords_raw = []
    for p in papers:
        if p["keywords"]:
            kws = re.split(r'[;；,，|]+', p["keywords"])
            all_keywords_raw.extend(kw.strip().lower() for kw in kws if kw.strip())
    keyword_freq = Counter(all_keywords_raw).most_common(100)
    stats["top_keywords"] = [{"keyword": kw, "count": cnt} for kw, cnt in keyword_freq]

    index = {
        "stats": stats,
        "papers": papers,
        "scanned_files": [p["filename"] for p in papers if p.get("is_scannable")],
        "error_files": [p["filename"] for p in papers if p.get("error")],
    }

    # 保存
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    json_size = os.path.getsize(OUTPUT_JSON)
    print(f"\n=== 构建完成 ===")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"索引文件: {OUTPUT_JSON} ({json_size/1024/1024:.1f} MB)")
    print(f"总论文: {stats['total_papers']} (本地{stats['local_papers']} + Zotero{stats['zotero_papers']})")
    print(f"有Zotero元数据: {stats['with_zotero_meta']}")
    print(f"有摘要: {stats['with_abstract']} ({stats['with_abstract']*100//max(stats['total_papers'],1)}%)")
    print(f"  其中兜底提取: {stats['abstract_with_fallback']}")
    print(f"有关键词: {stats['with_keywords']} ({stats['with_keywords']*100//max(stats['total_papers'],1)}%)")
    print(f"  其中自动生成: {stats['keywords_auto']}")
    print(f"中文: {stats['chinese_papers']}, 英文: {stats['english_papers']}")
    print(f"学位论文: {stats['thesis_papers']}")
    print(f"扫描件: {stats['scanned_papers']}")
    print(f"读取失败: {stats['error_papers']}")
    print(f"提取方法分布: {stats['by_method']}")

    if keyword_freq:
        print(f"\n=== Top 20 高频关键词 ===")
        for kw, cnt in keyword_freq[:20]:
            print(f"  {kw}: {cnt}")

    generate_readable_md(papers, stats)
    return index


def generate_readable_md(papers, stats):
    lines = []
    lines.append("# 文献索引库 v3")
    lines.append(f"\n构建时间: {stats['build_date']}")
    lines.append(f"总计: {stats['total_papers']}篇 (本地{stats['local_papers']} + Zotero{stats['zotero_papers']}) | "
                 f"有摘要: {stats['with_abstract']}篇 | "
                 f"有关键词: {stats['with_keywords']}篇 | "
                 f"中文: {stats['chinese_papers']}篇 | 英文: {stats['english_papers']}篇")
    lines.append(f"学位论文: {stats['thesis_papers']}篇 | 扫描件: {stats['scanned_papers']}篇\n")

    by_folder = defaultdict(list)
    for p in papers:
        by_folder[p["folder"]].append(p)

    for folder in sorted(by_folder.keys()):
        folder_papers = by_folder[folder]
        abs_count = sum(1 for p in folder_papers if p["abstract"])
        lines.append(f"\n## {folder} ({len(folder_papers)}篇, {abs_count}篇有摘要)\n")

        for p in folder_papers:
            name = p["filename"]
            year = f"[{p['year']}]" if p["year"] else ""
            lang = "中" if p["language"] == "zh" else "英" if p["language"] == "en" else "?"
            method = p.get("extraction_method", "")
            method_tag = f" 📖{method}" if method != "standard" else ""
            thesis_tag = " 🎓" if p.get("is_thesis") else ""
            source_tag = " 📚Z" if p.get("source") == "zotero" else ""

            if p.get("is_scannable"):
                lines.append(f"- **{name}** {year} ({lang}) ⚠️扫描件{source_tag}")
                continue

            lines.append(f"- **{name}** {year} ({lang}) {p['page_count']}页{thesis_tag}{method_tag}{source_tag}")

            if p["keywords"]:
                lines.append(f"  - 关键词: {p['keywords'][:300]}")
            if p["abstract"]:
                abs_short = p["abstract"][:400]
                if len(p["abstract"]) > 400:
                    abs_short += "..."
                lines.append(f"  - 摘要: {abs_short}")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    md_size = os.path.getsize(OUTPUT_MD)
    print(f"可读版本: {OUTPUT_MD} ({md_size/1024:.0f} KB)")


if __name__ == "__main__":
    incremental = "--incr" in sys.argv
    no_zotero = "--no-zotero" in sys.argv
    build_index(incremental=incremental, include_zotero=not no_zotero)
