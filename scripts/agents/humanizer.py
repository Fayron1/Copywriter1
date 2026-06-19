# agents/humanizer.py
"""
Статистический хуманайзер (0 токенов на анализ).

Идея:
  1. analyze_section() / analyze_article() считают ОБЪЕКТИВНЫЕ метрики человечности
     (burstiness, вариативность длины предложений, лексическое разнообразие,
     повтор зачинов, равномерность абзацев, штампы) — всё на чистом Python.
  2. select_sections_to_fix() возвращает ТОЛЬКО худшие секции (max_fix штук),
     отсортированные по возрастанию human_score. Точечная коррекция.
  3. humanize_article() принимает callback rewrite_fn (его даёт pipeline), переписывает
     выбранные секции ПО ОДНОМУ РАЗУ каждую, по стратегии accept-best:
       - если после переписи скор вырос → принимаем и ЗАМОРАЖИВАЕМ секцию;
       - если не вырос → откатываем к оригиналу.
     Глобального повторного цикла нет → перерасход токенов невозможен.

Контракт rewrite_fn:
    rewrite_fn(section_raw: str, instruction: str, prev_tail: str, next_head: str) -> str
    Должна вернуть переписанный markdown ТОЛЬКО этой секции (с заголовком).
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    from .stopwords import ALL_STOP_WORDS
except Exception:  # автономный запуск/тесты
    ALL_STOP_WORDS = []

# ─────────────────────────────────────────────────────────────────────────────
# Нормы для русской B2B-прозы (откалибровано под живой экспертный текст).
# Это «человеческий коридор», а не жёсткие пороги для одиночного предложения.
# ─────────────────────────────────────────────────────────────────────────────
SENT_STDEV_MIN = 5.5      # σ длины предложений (в словах); ниже → монотонно (ИИ)
SENT_CV_MIN = 0.45        # коэф. вариации stdev/mean; ниже → ровный «кирпич»
CLUSTER_MAX = 0.55        # доля предложений в ±3 слова от среднего; выше → робот
TTR_MIN = 0.42            # лексическое разнообразие (MATTR, окно 50)
START_REPEAT_MAX = 0.30   # доля предложений с самым частым словом-зачином
PARA_CV_MIN = 0.30        # вариативность длины абзацев; ниже → одинаковые блоки
CONNECTOR_MAX = 0.18      # доля предложений, начатых с вводных связок

# Вводные связки-зачины (их перебор выдаёт ИИ-ритм)
CONNECTORS = {
    "однако", "таким образом", "кроме того", "тем не менее", "более того",
    "следовательно", "соответственно", "вместе с тем", "в свою очередь",
    "при этом", "также", "итак", "впрочем", "напротив", "наконец",
}

# Сокращения, на которых НЕ разрываем предложение
_ABBR = {
    "т.д.", "т.е.", "т.п.", "т.к.", "и.о.", "г.", "гг.", "руб.", "коп.",
    "ст.", "стст.", "п.", "пп.", "ч.", "абз.", "млн.", "млрд.", "тыс.",
    "см.", "напр.", "др.", "пр.", "вкл.", "рис.", "табл.", "стр.",
}

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка текста
# ─────────────────────────────────────────────────────────────────────────────
def _strip_markdown_noise(text: str) -> str:
    """Убираем то, что не является прозой: заголовки, таблицы, код, маркеры списков."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):                      # заголовки
            continue
        if s.startswith("|") or re.match(r"^[:\-\|\s]+$", s):  # строки таблиц
            continue
        if s.startswith("```"):                    # код-фенсы
            continue
        if s.startswith(">"):                      # вырезки-цитаты оставляем как прозу
            s = s.lstrip("> ").strip()
        # снимаем маркеры списков, но сам пункт считаем предложением
        s = re.sub(r"^(\d+[\.\)]\s+|[-*•]\s+)", "", s)
        out.append(s)
    return "\n".join(out)


def split_sentences(text: str) -> List[str]:
    """Грубое, но устойчивое к рус. сокращениям деление на предложения."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # защищаем сокращения и десятичные числа от разрыва
    protected = text
    for i, ab in enumerate(_ABBR):
        protected = protected.replace(ab, f"\x00{i}\x00")
    protected = re.sub(r"(\d)\.(\d)", "\\1\x01\\2", protected)  # 1.5 -> неразрывное

    parts = re.split(r"(?<=[.!?…])\s+(?=[«\"A-ZА-ЯЁ0-9])", protected)

    sentences = []
    for p in parts:
        p = p.replace("\x01", ".")
        for i, ab in enumerate(_ABBR):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def _words(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def _mattr(tokens: List[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio — лексическое разнообразие без зависимости от длины."""
    if not tokens:
        return 1.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = []
    for i in range(len(tokens) - window + 1):
        win = tokens[i:i + window]
        ratios.append(len(set(win)) / window)
    return sum(ratios) / len(ratios)


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SectionMetrics:
    index: int
    heading: str
    char_len: int
    n_sentences: int
    mean_len: float
    stdev_len: float
    cv: float
    cluster_ratio: float
    ttr: float
    start_repeat_ratio: float
    para_cv: float
    connector_ratio: float
    stamp_hits: List[str]
    human_score: float
    issues: List[str] = field(default_factory=list)


def analyze_section(index: int, heading: str, raw: str) -> SectionMetrics:
    prose = _strip_markdown_noise(raw)
    sentences = split_sentences(prose)
    tokens = _words(prose)
    n = len(sentences)

    lengths = [len(_words(s)) for s in sentences] or [0]
    mean_len = statistics.mean(lengths) if lengths else 0.0
    stdev_len = statistics.pstdev(lengths) if n >= 2 else 0.0
    cv = (stdev_len / mean_len) if mean_len else 0.0

    # доля предложений в ±3 слова от среднего (кластеризация = монотонность)
    cluster_ratio = (sum(1 for L in lengths if abs(L - mean_len) <= 3) / n) if n else 0.0

    ttr = _mattr(tokens)

    # повтор слова-зачина
    starts = [_words(s)[0] for s in sentences if _words(s)]
    if starts:
        from collections import Counter
        most = Counter(starts).most_common(1)[0][1]
        start_repeat_ratio = most / len(starts)
    else:
        start_repeat_ratio = 0.0

    # связки-зачины
    connector_hits = 0
    for s in sentences:
        low = s.lower().lstrip("«\"")
        if any(low.startswith(c) for c in CONNECTORS):
            connector_hits += 1
    connector_ratio = (connector_hits / n) if n else 0.0

    # равномерность абзацев
    paras = [p for p in raw.split("\n\n") if p.strip()]
    para_lens = [len(p) for p in paras] or [0]
    para_mean = statistics.mean(para_lens) if para_lens else 0.0
    para_cv = (statistics.pstdev(para_lens) / para_mean) if (len(para_lens) >= 2 and para_mean) else 0.0

    # штампы
    low_all = prose.lower()
    stamp_hits = [w for w in ALL_STOP_WORDS if w in low_all]

    # ── скоринг human_score (0..100), штрафная модель ───────────────────────
    issues: List[str] = []
    score = 100.0

    # короткие секции (мало предложений) метрикам ритма не доверяем
    rhythm_reliable = n >= 4

    if rhythm_reliable:
        if stdev_len < SENT_STDEV_MIN:
            pen = min(28, (SENT_STDEV_MIN - stdev_len) * 5)
            score -= pen
            issues.append(
                f"монотонный ритм: σ длины предложений {stdev_len:.1f} (норма >{SENT_STDEV_MIN}); "
                f"средняя длина {mean_len:.0f} слов"
            )
        if cv < SENT_CV_MIN:
            score -= min(12, (SENT_CV_MIN - cv) * 30)
            issues.append(f"низкая вариативность длины (CV {cv:.2f} < {SENT_CV_MIN})")
        if cluster_ratio > CLUSTER_MAX:
            score -= min(15, (cluster_ratio - CLUSTER_MAX) * 40)
            issues.append(
                f"{cluster_ratio*100:.0f}% предложений почти одинаковой длины "
                f"(~{mean_len:.0f} слов) — добавь короткие (4-7 слов) и длинные (20+ слов)"
            )

    if ttr < TTR_MIN and len(tokens) >= 40:
        score -= min(15, (TTR_MIN - ttr) * 50)
        issues.append(f"бедный словарь (разнообразие {ttr:.2f} < {TTR_MIN}), много повторов слов")

    if start_repeat_ratio > START_REPEAT_MAX and n >= 5:
        score -= min(12, (start_repeat_ratio - START_REPEAT_MAX) * 40)
        issues.append(
            f"{start_repeat_ratio*100:.0f}% предложений начинаются с одного слова — разнообразь зачины"
        )

    if connector_ratio > CONNECTOR_MAX and n >= 5:
        score -= min(10, (connector_ratio - CONNECTOR_MAX) * 40)
        issues.append(f"перебор вводных связок в зачинах ({connector_ratio*100:.0f}%)")

    if para_cv and para_cv < PARA_CV_MIN and len(paras) >= 3:
        score -= min(8, (PARA_CV_MIN - para_cv) * 25)
        issues.append("абзацы одинаковой длины («кирпичи») — сделай их разной длины")

    if stamp_hits:
        score -= min(20, len(stamp_hits) * 6)
        issues.append(f"ИИ-штампы: {', '.join(stamp_hits[:5])}")

    score = max(0.0, min(100.0, score))

    return SectionMetrics(
        index=index, heading=heading or "(вступление)", char_len=len(raw),
        n_sentences=n, mean_len=round(mean_len, 1), stdev_len=round(stdev_len, 2),
        cv=round(cv, 2), cluster_ratio=round(cluster_ratio, 2), ttr=round(ttr, 2),
        start_repeat_ratio=round(start_repeat_ratio, 2), para_cv=round(para_cv, 2),
        connector_ratio=round(connector_ratio, 2), stamp_hits=stamp_hits,
        human_score=round(score, 1), issues=issues,
    )


def analyze_article(sections: List[Dict]) -> Dict:
    """sections — вывод _split_markdown_sections (ключи level/heading/raw)."""
    per = []
    for i, b in enumerate(sections):
        per.append(analyze_section(i, b.get("heading", ""), b.get("raw", "")))

    scored = [m for m in per if m.n_sentences >= 3]  # короткие секции в средний скор не берём
    article_score = round(statistics.mean([m.human_score for m in scored]), 1) if scored else 100.0
    return {"article_human_score": article_score, "sections": per}


# ─────────────────────────────────────────────────────────────────────────────
# Выбор секций для точечной правки
# ─────────────────────────────────────────────────────────────────────────────
def select_sections_to_fix(
    sections: List[Dict],
    metrics: List[SectionMetrics],
    *,
    min_score: float = 70.0,
    max_fix: int = 3,
    editable_levels: tuple = (2, 3),
    frozen: Optional[set] = None,
) -> List[SectionMetrics]:
    """Возвращает не более max_fix худших РЕДАКТИРУЕМЫХ секций со скором ниже min_score."""
    frozen = frozen or set()
    cands = []
    for m in metrics:
        if m.index in frozen:
            continue
        level = sections[m.index].get("level", 0)
        if level not in editable_levels:
            continue
        if m.n_sentences < 4:            # слишком короткая — нечего «очеловечивать»
            continue
        if m.human_score >= min_score:   # уже норм
            continue
        cands.append(m)
    cands.sort(key=lambda m: m.human_score)   # сначала самые провальные
    return cands[:max_fix]


def build_instruction(m: SectionMetrics) -> str:
    """Конкретная инструкция для rewrite_fn: что чинить, с реальными числами."""
    head = (
        "Перепиши ТОЛЬКО эту секцию, чтобы сломать машинный ритм. "
        "Сохрани ВСЕ факты, цифры, номера статей законов, заголовок и структуру списков/таблиц. "
        "Не добавляй и не удаляй смысл — меняй только форму подачи.\n"
        "Конкретные проблемы этой секции:\n"
    )
    body = "\n".join(f"- {i}" for i in m.issues) or "- сделай ритм живее"
    tail = (
        "\nЦель: σ длины предложений выше "
        f"{SENT_STDEV_MIN} (чередуй короткие хлёсткие фразы 4-7 слов с развёрнутыми 20+ слов), "
        "разные зачины, без штампов. Верни только markdown этой секции."
    )
    return head + body + tail


# ─────────────────────────────────────────────────────────────────────────────
# Оркестрация: точечная правка, accept-best, заморозка, жёсткий лимит
# ─────────────────────────────────────────────────────────────────────────────
def humanize_article(
    full_text: str,
    split_fn: Callable[[str], List[Dict]],
    reassemble_fn: Callable[[List[Dict]], str],
    rewrite_fn: Callable[[str, str, str, str], str],
    *,
    min_score: float = 70.0,
    max_fix: int = 3,
    editable_levels: tuple = (2, 3),
    logger=None,
) -> Dict:
    """
    Один проход. Анализ — 0 токенов. LLM-вызовов = ровно len(выбранных секций) ≤ max_fix.
    Каждая секция переписывается максимум ОДИН раз (accept-best + заморозка).
    """
    def log(msg):
        if logger:
            logger.info(msg)

    sections = split_fn(full_text)
    pre = analyze_article(sections)
    log(f"🧪 Human-score статьи (до): {pre['article_human_score']}/100")

    targets = select_sections_to_fix(
        sections, pre["sections"], min_score=min_score, max_fix=max_fix,
        editable_levels=editable_levels,
    )
    if not targets:
        log("✅ Хуманайзер: все секции выше порога, правки не нужны.")
        return {"text": full_text, "score_before": pre["article_human_score"],
                "score_after": pre["article_human_score"], "rewrites": 0, "details": []}

    log(f"✍️ Хуманайзер: к точечной правке выбрано {len(targets)} секц. "
        f"(индексы { [t.index for t in targets] }, лимит {max_fix})")

    new_sections = list(sections)
    details = []
    rewrites = 0

    for m in targets:
        idx = m.index
        sec = sections[idx]
        prev_tail = sections[idx - 1]["raw"].strip()[-300:] if idx - 1 >= 0 else ""
        next_head = sections[idx + 1]["raw"].strip()[:300] if idx + 1 < len(sections) else ""
        instruction = build_instruction(m)

        try:
            rewritten = rewrite_fn(sec["raw"], instruction, prev_tail, next_head)
        except Exception as e:
            log(f"   ⚠️ Секция [{idx}] '{m.heading[:40]}': ошибка переписи ({e}). Пропуск.")
            continue
        rewrites += 1

        # accept-best: принимаем ТОЛЬКО если стало человечнее и длина адекватна
        new_m = analyze_section(idx, sec.get("heading", ""), rewritten or "")
        len_ok = rewritten and len(rewritten) >= len(sec["raw"]) * 0.6
        if len_ok and new_m.human_score > m.human_score:
            trailing = sec["raw"][len(sec["raw"].rstrip()):]
            new_sections[idx] = {**sec, "raw": rewritten.rstrip() + trailing}
            log(f"   🩹 Секция [{idx}] '{m.heading[:40]}': {m.human_score}→{new_m.human_score}")
            details.append({"index": idx, "before": m.human_score, "after": new_m.human_score,
                            "accepted": True})
        else:
            log(f"   ↩️ Секция [{idx}] '{m.heading[:40]}': без улучшения "
                f"({m.human_score}→{new_m.human_score}), откат.")
            details.append({"index": idx, "before": m.human_score, "after": new_m.human_score,
                            "accepted": False})
        # секция в любом случае больше не трогается (заморожена) — цикла нет

    new_text = reassemble_fn(new_sections)
    post = analyze_article(split_fn(new_text))
    log(f"🏁 Human-score статьи (после): {post['article_human_score']}/100 | переписей: {rewrites}")

    return {"text": new_text, "score_before": pre["article_human_score"],
            "score_after": post["article_human_score"], "rewrites": rewrites, "details": details}
