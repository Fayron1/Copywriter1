"""
Deliverables Renderer — превращает финальную статью в контент-пакет.

Пакет «Экспертная статья под публикацию»:
  - seo_title / meta_description (LLM: packager на дешёвой модели)
  - FAQ (детерминированно: разбор существующего раздела FAQ статьи;
    если его нет — packager генерирует по H2)
  - 3 Telegram-поста (LLM: три разных угла — боль, цифра, дайджест)
  - короткий анонс (LLM)
  - source_passport (детерминированно: claim pack + источники из state)
  - assumptions_and_limits (детерминированно: limitations claim'ов)
  - cta_block (детерминированно: cta_renderer)

Промпты для packager запрещают абсолюты, псевдо-практику и промпт-лик —
вывод проходит lint_legal перед включением в пакет.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("agents.deliverables")

# ────────────────────────────────────────────────────────────
# Детерминированные части
# ────────────────────────────────────────────────────────────

_H1_RE = re.compile(r"^#\s+(.+)$", re.M)
_H2_FAQ_RE = re.compile(r"^##\s*(?:FAQ|Частые вопросы|Вопросы и ответы)\s*$", re.I | re.M)
_Q_RE = re.compile(r"^(?:\*\*)?(?:\d+[.)]\s*)?([А-ЯЁ][^?\n]{5,120}\?)", re.M)


def _strip_frontmatter(article: str) -> str:
    """Убрать YAML-шапку (--- title: ... ---) — не контент статьи."""
    if article.startswith("---"):
        end = article.find("\n---", 3)
        if end != -1:
            return article[end + 4:].lstrip("\n")
    return article


def extract_h1(article: str) -> str:
    m = _H1_RE.search(article or "")
    if m:
        return m.group(1).strip()
    return (article or "").strip().split("\n", 1)[0][:100]


def build_seo_title(article: str, fallback_topic: str = "") -> str:
    """Заголовок для SERP: ≤65 знаков, без обрыва слова."""
    base = extract_h1(article) or fallback_topic
    base = re.sub(r"\s*[—:|].*$", "", base).strip()  # хвост после тире режем
    if len(base) <= 65:
        return base
    cut = base[:65].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.—")


def extract_faq(article: str) -> List[Dict[str, str]]:
    """Разобрать раздел FAQ статьи в [{q, a}]. Пусто → []."""
    if not article:
        return []
    m = _H2_FAQ_RE.search(article)
    if not m:
        return []
    tail = article[m.end():]
    # до следующего H2
    nxt = re.search(r"^##\s", tail, re.M)
    if nxt:
        tail = tail[:nxt.start()]
    items = []
    # Вопрос — жирная строка или строка с ?, ответ — следующий абзац
    parts = re.split(r"\n(?=(?:\*\*[^*]+\*\*|[\d]+[.)]\s*[А-ЯЁ]))", tail)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        qm = re.match(r"(?:\*\*)?((?:\d+[.)]\s*)?[^*\n]{5,140}\?)(?:\*\*)?", p)
        if not qm:
            continue
        q = qm.group(1).strip()
        a = p[qm.end(0):].strip().lstrip("—-: ").strip()
        if q and a:
            items.append({"q": q, "a": re.sub(r"\s+", " ", a)[:500]})
    return items


def build_source_passport(claims: List[Dict], references: List[Dict],
                          topic: str, article_year: int = 2026) -> Dict:
    """Паспорт фактов: approved claims + ограничения + источники."""
    approved = [c for c in (claims or []) if c.get("status", "").startswith("supported")]
    limitations = []
    for c in (claims or []):
        lim = c.get("limitations") or c.get("limitation")
        if lim and lim not in limitations:
            limitations.append(lim)
    srcs = [
        {"title": (r.get("title") or "")[:120], "url": r.get("url", "")}
        for r in (references or []) if r.get("url") or r.get("title")
    ]
    return {
        "topic": topic,
        "verify_year": article_year,
        "as_of_date": datetime.date.today().strftime("%Y-%m-%d"),
        "procurement_law": _detect_procurement_law(topic),
        "procedure": _detect_procedure(topic),
        "high_risk_claims": [
            # ФИКС (аудит 🔴8): claims пайплайна несут claim_type/text,
            # ledger — type/claim; читаем оба контракта
            ((c.get("claim_type") and c.get("text")) or c.get("claim") or "")[:100]
            for c in (claims or [])
            if (c.get("claim_type") or c.get("type") or "")
            in ("legal_rule", "calculation")
            and (c.get("risk") or "").lower() in ("high", "red")
        ][:10],
        "approved_claims": len(approved),
        "total_claims": len(claims or []),
        "limitations": limitations[:10],
        "sources": srcs[:10],
        "note": (
            "Финальную приёмку правовых выводов проводит профильный "
            "эксперт перед публикацией."
        ),
    }


# Детект закупочного контура и процедуры (расширенная система правил,
# раздел «Иерархия применимости» — упрощённо, по теме статьи)
_PROC_LAW_RX_44 = re.compile(r"44[-\s]?ФЗ|госзакуп|контрактн\w+ систем", re.I)
_PROC_LAW_RX_223 = re.compile(r"223[-\s]?ФЗ", re.I)
_PROCEDURES = [
    ("электронный аукцион", r"аукцион"),
    ("электронный конкурс", r"конкурс"),
    ("электронный запрос котировок", r"котировок"),
    ("закупка у единственного поставщика", r"единственн\w+\s+поставщик"),
    ("закупка у МСП", r"\bМСП\b|малого\s+предпринимательств"),
    ("реестр недобросовестных поставщиков", r"\bРНП\b|недобросовестн\w+ поставщик"),
    ("обжалование в ФАС", r"жалоб\w+|обжалова\w+"),
]


def _detect_procurement_law(topic: str) -> str:
    t = topic or ""
    has44, has223 = bool(_PROC_LAW_RX_44.search(t)), bool(_PROC_LAW_RX_223.search(t))
    if has44 and has223:
        return "44-ФЗ + 223-ФЗ (сравнение)"
    if has223:
        return "223-ФЗ"
    if has44:
        return "44-ФЗ"
    return ""


def _detect_procedure(topic: str) -> str:
    t = topic or ""
    for name, rx in _PROCEDURES:
        if re.search(rx, t, re.I):
            return name
    return ""


_META_NUMBER_RX = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|млн|млрд|тыс|дн\w*|дней|"
                             r"дня|руб|₽|час\w*)", re.I)


def _meta_claims_in_body(meta: str, body: str) -> bool:
    """Гвард packager (кейс статьи 24v2): мета обещала «10 дней на возврат
    обеспечения» — нормы нет ни в 44-ФЗ, ни в теле статьи. Числовые
    заявления мета-описания обязаны существовать в теле."""
    if not meta or not body:
        return True
    for claim in _META_NUMBER_RX.findall(meta):
        # нормализуем «5 дн.» / «5 дней» / «30 дней» — ищем число + единицу
        num = re.match(r"\d+(?:[.,]\d+)?", claim).group(0).replace(",", ".")
        unit = claim[len(num):].strip().lower()
        unit_stem = unit[:3]
        body_l = body.lower()
        if not re.search(re.escape(num) + r"[^\d]{0,3}" + re.escape(unit_stem),
                         body_l):
            return False
    return True


# ────────────────────────────────────────────────────────────
# LLM-части (packager — дешёвая модель)
# ────────────────────────────────────────────────────────────

_META_PROMPT = """Ты — SEO-специалист. По статье ниже верни СТРОГО JSON:
{"meta_description": "..."}
Требования к meta_description: 140–160 знаков, включает главный факт
(цифру/срок/ставку) из статьи, без клише «в статье мы рассмотрим»,
без кавычек внутри. Только JSON, без markdown.

Статья (фрагмент):
@@EXCERPT@@"""

_FAQ_PROMPT = """По статье ниже составь 5 вопросов-ответов для раздела FAQ
блога B2B-компании. Формат СТРОГО JSON-массив:
[{"q": "...", "a": "..."}]
Вопросы — реальные возражения читателя (сроки, цифры, «а если...»).
Ответы 1–3 предложения, только факты из статьи, без выдумки.
Без абсолютов («лучший», «гарантированно»), без обращения «вы» с поучениями.

Статья:
@@EXCERPT@@"""

_TG_PROMPT = """По статье ниже напиши 3 поста для Telegram-канала B2B-аудитории.
Верни СТРОГО JSON-массив из 3 объектов:
[{"angle": "pain", "text": "..."}, {"angle": "number", "text": "..."}, {"angle": "digest", "text": "..."}]
- pain: боль читателя → главный вывод статьи (2–4 предложения)
- number: один яркий факт/цифру статьи как крючок (2–3 предложения)
- digest: микродайджест 3 пунктов («—» на строку)
Правила: каждый пост ≤ 700 знаков; без хэштегов; без «ссылка в первом
комментарии»; без абсолютов и обещаний; тон спокойный экспертный;
без эмодзи, кроме максимум одного на пост.

Статья:
@@EXCERPT@@"""

_ANNOUNCE_PROMPT = """По статье ниже напиши короткий анонс (2 предложения,
≤ 280 знаков) для соцсетей и рассылки: что читатель узнаёт и зачем ему это.
Без клише и без призывов «читайте в статье». Верни только текст анонса.

Статья (фрагмент):
@@EXCERPT@@"""

_EXCERPT_CHARS = 9000


def _fill(prompt: str, excerpt: str) -> str:
    """Подстановка фрагмента статьи. Без str.format: JSON-скобки в
    промптах остаются буквальными, @@EXCERPT@@ — единственный маркер."""
    return prompt.replace("@@EXCERPT@@", excerpt)


def _call_packager(prompt: str, llm_call: Callable[[str], str],
                   expect_json: bool = True) -> Optional[str]:
    """Вызов packager с валидацией. llm_call(prompt) -> text."""
    try:
        raw = (llm_call(prompt) or "").strip()
    except Exception as e:
        logger.warning(f"packager call failed: {e}")
        return None
    if not raw:
        return None
    if expect_json:
        # срезать markdown-обёртку ```json ... ```
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
    return raw


def _parse_json(raw: Optional[str], default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        # иногда модель добавляет текст вокруг JSON — берём {...}/[...]
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return default


def build_content_package(article: str, topic: str, direction: str,
                          claims: Optional[List[Dict]],
                          references: Optional[List[Dict]],
                          article_year: int = 2026,
                          llm_call: Optional[Callable[[str], str]] = None,
                          cta_mode: str = "self_promo") -> Dict:
    """Собрать контент-пакет. llm_call=None → только детерминированные части."""
    from .cta_renderer import render_cta
    from .norm_params import lint_legal

    article = _strip_frontmatter(article or "")
    excerpt = article[:_EXCERPT_CHARS]
    pkg: Dict = {
        "topic": topic,
        "direction": direction,
        "article_year": article_year,
    }

    # Детерминированное
    pkg["seo_title"] = build_seo_title(article, topic)
    # Ревью 24 (H1_PROMISE_NOT_COVERED_BY_BODY): цифры в заголовках обязаны
    # существовать в теле. Если seo_title обещает срок/сумму, которой в
    # статье нет — падаем на чистый заголовок статьи.
    if not _meta_claims_in_body(pkg["seo_title"], article):
        h1 = re.search(r"^#\s+(.+)$", article, re.M)
        pkg["seo_title"] = (h1.group(1).strip() if h1 else topic)[:70]
    pkg["source_passport"] = build_source_passport(claims or [], references or [],
                                                   topic, article_year)
    faq = extract_faq(article)
    pkg["faq_from_article"] = bool(faq)

    # LLM-части
    if llm_call and excerpt:
        raw_meta = _call_packager(_fill(_META_PROMPT, excerpt[:4000]), llm_call)
        meta = _parse_json(raw_meta, {})
        md = (meta or {}).get("meta_description", "")
        if 100 <= len(md) <= 200 and not lint_legal(md) \
                and _meta_claims_in_body(md, article):
            pkg["meta_description"] = md
        else:
            lead = re.sub(r"\s+", " ", re.sub(r"^#+.*$", "", excerpt)).strip()
            pkg["meta_description"] = lead[:157] + ("…" if len(lead) > 157 else "")

        if not faq:
            raw_faq = _call_packager(_fill(_FAQ_PROMPT, excerpt), llm_call)
            faq = _parse_json(raw_faq, [])
            kept_faq = []
            for f in faq:
                if not (isinstance(f, dict) and f.get("q") and f.get("a")):
                    continue
                _iss = lint_legal(f"{f['q']} {f['a']}")
                if _iss:
                    logger.warning(f"   🟡 пакет: FAQ отброшен линтером: "
                                   f"{str(_iss[0])[:100]} | {f['q'][:60]}")
                    continue
                kept_faq.append(f)
            faq = kept_faq[:5]
        raw_tg = _call_packager(_fill(_TG_PROMPT, excerpt), llm_call)
        tg = _parse_json(raw_tg, [])
        posts = []
        for p in tg if isinstance(tg, list) else []:
            if not (isinstance(p, dict) and p.get("text")
                    and len(p["text"]) <= 800):
                continue
            _iss = lint_legal(p["text"])
            if _iss:
                logger.warning(f"   🟡 пакет: TG-пост отброшен линтером: "
                               f"{str(_iss[0])[:100]}")
                continue
            posts.append({"angle": p.get("angle", ""), "text": p["text"]})
        pkg["telegram_posts"] = posts[:3]

        raw_ann = _call_packager(_fill(_ANNOUNCE_PROMPT, excerpt[:3000]),
                                 llm_call, expect_json=False)
        ann = (raw_ann or "").strip().strip('"')
        _iss_ann = lint_legal(ann) if ann else []
        if ann and len(ann) <= 400 and not _iss_ann:
            pkg["short_announcement"] = ann
        elif ann:
            logger.warning(f"   🟡 пакет: анонс отброшен линтером: "
                           f"{str(_iss_ann[0])[:100]}")

    pkg["faq"] = faq[:5]
    pkg["assumptions_and_limits"] = pkg["source_passport"]["limitations"]
    pkg["cta_block"] = render_cta(direction, cta_mode, topic_hint=topic)
    return pkg


# ── Green/Yellow/Red: домен → уровень риска поставки ──
TIER_BY_DIRECTION = {
    "налоги": "red",
    "трудовое": "red",
    "защита данных": "red",
    "закупки": "red",
    "бизнес": "yellow",
    "финансы": "yellow",
    "маркетинг": "green",
}

REVIEW_SCHEDULES = {
    "red": {
        "next_review": "январь 2027 (начало нового налогового периода)",
        "triggers": [
            "изменение ставки, лимита или формы отчётности",
            "новый закон или письмо ФНС/Минтруда/РКН по теме",
            "утверждение коэффициентов на следующий год",
        ],
    },
    "yellow": {
        "next_review": "квартал после публикации",
        "triggers": ["изменение регулирующих требований", "практика применения"],
    },
    "green": {
        "next_review": "через 6 месяцев",
        "triggers": ["сдвиг рынка или платформ", "изменение алгоритмов/форматов"],
    },
}


def build_expert_signoff(pkg: dict) -> str:
    """Чек-лист для эксперта клиента (Red-домены)."""
    sp = pkg.get("source_passport") or {}
    lines = [
        "# Чек-лист экспертной приёмки (обязателен до публикации)",
        "",
        f"Тема: {pkg.get('topic', '')}",
        f"Уровень риска домена: {pkg.get('tier', 'red').upper()}",
        f"Утверждённых утверждений в claim ledger: "
        f"{sp.get('approved_claims', 0)} из {sp.get('total_claims', 0)}",
        "",
        "Эксперту проверить:",
        "1. Все числовые значения (ставки, лимиты, сроки) против действующих редакций.",
        "2. Применимость норм к субъекту статьи (ИП/ООО, режим, отрасль).",
        "3. Процедурные утверждения (кто решает, в каком порядке).",
        "4. Расчётные примеры: допущения и арифметику.",
        "",
        "Подпись эксперта / дата: ____________",
    ]
    return "\n".join(lines)


def package_to_markdown(pkg: Dict) -> str:
    """Пакет → поставляемый markdown-файл."""
    lines = [f"# Контент-пакет: {pkg.get('topic', '')}", ""]
    lines += [f"- SEO title: {pkg.get('seo_title', '')}",
              f"- Meta description: {pkg.get('meta_description', '')}", ""]

    lines += ["## FAQ", ""]
    for f in pkg.get("faq", []):
        lines += [f"**{f['q']}**", f"{f['a']}", ""]

    if pkg.get("telegram_posts"):
        lines += ["## Telegram-посты", ""]
        for i, p in enumerate(pkg["telegram_posts"], 1):
            lines += [f"### Пост {i} ({p.get('angle', '')})", "", p["text"], ""]

    if pkg.get("short_announcement"):
        lines += ["## Анонс", "", pkg["short_announcement"], ""]

    sp = pkg.get("source_passport") or {}
    lines += ["## Паспорт источников", ""]
    lines += [f"- Утверждённых утверждений: {sp.get('approved_claims', 0)} "
              f"из {sp.get('total_claims', 0)}",
              f"- Год сверки: {sp.get('verify_year', '')}",
              f"- {sp.get('note', '')}", ""]
    if sp.get("limitations"):
        lines += ["Допущения и ограничения:", ""]
        lines += [f"- {l}" for l in sp["limitations"]] + [""]
    if sp.get("sources"):
        lines += ["Источники:", ""]
        lines += [f"- {s['title']}" + (f" — {s['url']}" if s.get("url") else "")
                  for s in sp["sources"]] + [""]

    if pkg.get("cta_block"):
        lines += ["## CTA-блок (уже в статье)", "", pkg["cta_block"], ""]

    # review_schedule: подписка на актуализацию — коммерческий слой поставки
    tier = pkg.get("tier", "green")
    sched = REVIEW_SCHEDULES.get(tier, REVIEW_SCHEDULES["green"])
    lines += ["## Плановая актуализация", "",
              f"- Следующий пересмотр: {sched['next_review']}",
              "- Триггеры досрочного пересмотра:"]
    lines += [f"  — {t}" for t in sched["triggers"]]
    if tier == "red":
        lines += ["", "Финальную приёмку правовых выводов проводит эксперт "
                      "клиента (см. expert_signoff_checklist.md)."]
    return "\n".join(lines)
