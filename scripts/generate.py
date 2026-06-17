#!/usr/bin/env python3
"""
generate.py — CLI для генерации статей через мультиагентный pipeline.

Использование:
    python generate.py "Как открыть ООО в 2026 году" --type seo --dir юридическое
    python generate.py "10 ошибок при УСН" --type seo --style checklist --dir налоги --no-scout
    python generate.py "Разбор налоговой реформы 2026" --type longread --dir налоги

Результат сохраняется в output/<timestamp>_<slug>/
"""
import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Загрузка .env
try:
    from dotenv import load_dotenv
    # Ищем .env на уровень выше (корень проекта)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate")


def slugify(text: str, max_len: int = 40) -> str:
    """Транслитерация и создание slug для имени папки."""
    # Простая транслитерация
    translit = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
        "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
        "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
    result = []
    for char in text.lower():
        result.append(translit.get(char, char))
    slug = "".join(result)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:max_len]


def _normalize_article_markdown(text: str) -> str:
    """Восстановить блочную структуру Markdown перед конвертацией в HTML.

    Агенты-редакторы нередко возвращают текст, где пустые строки между блоками
    схлопнуты в один перевод строки (а то и вовсе склеены). Без пустых строк
    парсер Markdown не распознаёт заголовки/списки и выдаёт «сплошной текст».
    Здесь мы гарантируем пустую строку до и после каждого заголовка и перед
    началом списка, а также схлопываем лишние переводы строк.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    def _is_heading(line: str) -> bool:
        return bool(re.match(r'^\s{0,3}#{1,6}\s', line))

    def _is_list(line: str) -> bool:
        return bool(re.match(r'^\s*(?:[-*+]|\d+[.)])\s', line))

    def _is_blank(line: str) -> bool:
        return line.strip() == ""

    out: list = []
    for line in text.split("\n"):
        if _is_heading(line):
            if out and not _is_blank(out[-1]):
                out.append("")
            out.append(line.strip())
            out.append("")
        elif _is_list(line):
            if out and not _is_blank(out[-1]) and not _is_list(out[-1]):
                out.append("")
            out.append(line)
        else:
            out.append(line)

    normalized = "\n".join(out)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


def save_html_preview(state, output_dir: Path):
    """Сгенерировать красивый, адаптивный HTML-просмотрщик статьи (дизайн 'НЕЙРОЦЕХ / Журнальный разворот')."""
    article_content = state.final_article or state.draft or ""
    if not article_content:
        return

    # Нормализуем переводы строк для совместимости Windows/Linux и надежной работы парсера
    article_content = article_content.replace("\r\n", "\n")

    # 1. Очистка от вводных преамбул LLM (например, "Вот статья...", "Конечно, я переписал...")
    # Удаляем любой текст до первого заголовка # или ##, если он похож на преамбулу
    first_header_pos = article_content.find('#')
    if first_header_pos > 0:
        preamble = article_content[:first_header_pos].strip()
        # Если преамбула не содержит разметки и похожа на разговорный текст, отрезаем её
        if len(preamble) < 500 and not any(m in preamble for m in ["##", "###", "!["]):
            logger.info(f"   🧹 save_html_preview: Удалена преамбула ИИ: '{preamble[:60]}...'")
            article_content = article_content[first_header_pos:]

    # 2. Очистка от неразрешенных текстовых маркеров картинок вида [картинка: ...] или [IMAGE_PROMPT_HERE]
    marker_pattern = r"\[(?:картинка|IMAGE_PROMPT_HERE)(?::\s*.*?)?\]"
    article_content = re.sub(marker_pattern, "", article_content)

    # 3. Восстанавливаем блочную структуру Markdown (заголовки/списки)
    article_content = _normalize_article_markdown(article_content)

    # Определение человеческого типа статьи
    type_names = {
        "seo": "СЕО-статья",
        "longread": "Лонгрид",
    }
    style_names = {
        "checklist": "Чек-лист «10 пунктов»",
        "analysis": "Аналитический лонгрид",
        "case_study": "Бизнес-кейс",
        "law_review": "Разбор законодательства",
        "reference": "Практический справочник",
        "free_style": "Свободная статья",
    }
    type_name = style_names.get(getattr(state, "style_id", ""), type_names.get(state.article_type, "Статья"))
    direction = state.direction or "Бизнес"
    
    # Встроенный парсер метаданных для полной независимости от pipeline.py
    def _extract_meta_from_text(text: str) -> dict:
        meta_dict = {}
        if not text:
            return meta_dict
        title_m = re.search(r'(?:title|заголовок|meta\s+title)\s*:\s*["\'«]?(.*?)["\'»]?(?:\n|$)', text, re.IGNORECASE)
        if title_m:
            meta_dict['title'] = title_m.group(1).strip()
        desc_m = re.search(r'(?:description|описание|meta\s+description)\s*:\s*["\'«]?(.*?)["\'»]?(?:\n|$)', text, re.IGNORECASE)
        if desc_m:
            meta_dict['description'] = desc_m.group(1).strip()
        kw_m = re.search(r'(?:keywords|ключевые\s+слова)\s*:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if kw_m:
            kw_content = kw_m.group(1).strip().strip("[]\"'")
            meta_dict['keywords'] = [k.strip().strip('"\'-') for k in kw_content.split(',') if k.strip()]
        return meta_dict
    
    meta = state.final_meta or {}
    meta_title = meta.get("title")
    meta_description = meta.get("description")
    keywords_list = meta.get("keywords") or []

    # Если метаданных не хватает в final_meta, пробуем спасти их с помощью regex из сырого ответа Booster или текста статьи
    if not meta_title or not meta_description or not keywords_list:
        logger.info("   🔍 HTML Preview: Запуск регулярного парсера для извлечения метаданных...")
        raw_booster = state.seo_package.get("raw_response", "") if state.seo_package else ""
        extracted = _extract_meta_from_text(raw_booster) if raw_booster else {}
        
        extracted_article = _extract_meta_from_text(article_content)
        for k, v in extracted_article.items():
            if v and not extracted.get(k):
                extracted[k] = v

        if not meta_title and extracted.get("title"):
            meta_title = extracted["title"]
        if not meta_description and extracted.get("description"):
            meta_description = extracted["description"]
        if not keywords_list and extracted.get("keywords"):
            keywords_list = extracted["keywords"]

    # Фолбеки при полном отсутствии
    if not meta_title:
        meta_title = state.topic
    if not meta_description:
        meta_description = "Статья сгенерирована мультиагентной системой Copywriter."

    # Нормализуем формат ключевых слов
    if isinstance(keywords_list, str):
        keywords_list = [k.strip() for k in keywords_list.split(",") if k.strip()]

    keywords_html = "".join([
        f'<span class="keyword-tag">{kw}</span>'
        for kw in keywords_list
    ])

    # Преобразуем Markdown в HTML
    html_body = ""
    try:
        import markdown
        # Официальный парсер. Без 'nl2br' (он превращал одиночные переводы строк
        # в <br> и ломал структуру); 'sane_lists' — корректные списки.
        html_body = markdown.markdown(
            article_content,
            extensions=['tables', 'fenced_code', 'sane_lists'],
        )
    except ImportError:
        logger.warning("⚠️ Библиотека 'markdown' не установлена. Используется встроенный упрощенный парсер. Для идеального рендеринга таблиц и списков рекомендуется выполнить: pip install markdown")
        
        # Простой встроенный регулярный парсер (чтобы работало без пипа)
        html = article_content
        html = html.replace("<", "&lt;").replace(">", "&gt;")
        
        html = re.sub(r'^#\s+(.*?)$', r'<h1 class="text-3xl font-bold my-6">\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^##\s+(.*?)$', r'<h2 class="text-2xl font-bold my-4 border-b pb-2">\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^###\s+(.*?)$', r'<h3 class="text-xl font-bold my-3">\1</h3>', html, flags=re.MULTILINE)
        
        html = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1" />', html)
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        html = re.sub(r'^&gt;\s+(.*?)$', r'<blockquote class="border-l-4 border-primary pl-4 my-4 italic text-textMuted">\1</blockquote>', html, flags=re.MULTILINE)
        
        # Преобразуем маркированные списки в <li>
        html = re.sub(r'^[ \t]*[-*][ \t]+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        
        # Оборачиваем группы <li> в <ul>
        def wrap_ul(match):
            return '<ul>\n' + match.group(0) + '\n</ul>'
        html = re.sub(r'(?:<li>.*?</li>\n?)+', wrap_ul, html)
        
        # Преобразуем нумерованные списки
        html = re.sub(r'^[ \t]*\d+\.[ \t]+(.*?)$', r'<li-ord>\1</li-ord>', html, flags=re.MULTILINE)
        
        # Оборачиваем группы <li-ord> в <ol>
        def wrap_ol(match):
            items = match.group(0).replace('<li-ord>', '<li>').replace('</li-ord>', '</li>')
            return '<ol>\n' + items + '\n</ol>'
        html = re.sub(r'(?:<li-ord>.*?</li-ord>\n?)+', wrap_ol, html)
        
        # Парсинг таблиц Markdown
        def parse_tables(text_content):
            lines = text_content.split('\n')
            output_lines = []
            in_table = False
            table_html = []
            for line in lines:
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    if not in_table:
                        in_table = True
                        table_html.append('<table>')
                        table_html.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead>')
                        table_html.append('<tbody>')
                    else:
                        if all(re.match(r'^[-:\s]+$', c) for c in cells):
                            continue
                        table_html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
                else:
                    if in_table:
                        in_table = False
                        table_html.append('</tbody></table>')
                        output_lines.append('\n'.join(table_html))
                        table_html = []
                    output_lines.append(line)
            if in_table:
                table_html.append('</tbody></table>')
                output_lines.append('\n'.join(table_html))
            return '\n'.join(output_lines)
            
        html = parse_tables(html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        paragraphs = html.split('\n\n')
        for i, p in enumerate(paragraphs):
            p_strip = p.strip()
            if not p_strip:
                continue
            if not p_strip.startswith('<h') and not p_strip.startswith('<blockquote') and not p_strip.startswith('<ul') and not p_strip.startswith('<ol') and not p_strip.startswith('<li') and not p_strip.startswith('<img') and not p_strip.startswith('<table'):
                paragraphs[i] = f'<p class="mb-4 text-justify leading-relaxed">{p_strip}</p>'
        html_body = "\n\n".join(paragraphs)
        
    html_body = re.sub(
        r'<img([^>]*?)alt="Обложка"([^>]*?)>',
        r'<img\1alt="Обложка" class="cover-image"\2>',
        html_body,
        flags=re.IGNORECASE
    )
    html_body = re.sub(
        r'<img([^>]*?)alt="Иллюстрация"([^>]*?)>',
        r'<img\1alt="Иллюстрация" class="section-image"\2>',
        html_body,
        flags=re.IGNORECASE
    )

    # Кастомная подсветка блоков Решение, Ошибка, ВАЖНО с премиальными иконками и стилями
    # Решение:
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<p>\s*<strong>Решение:</strong>(.*?)</p>\s*</blockquote>',
        r'<div class="quote-block acceptance-block"><div class="quote-content"><div class="checklist-icon acceptance-icon"><i class="fa-solid fa-check"></i></div><div class="quote-text"><p><strong>Решение:</strong>\1</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<strong>Решение:</strong>(.*?)\s*</blockquote>',
        r'<div class="quote-block acceptance-block"><div class="quote-content"><div class="checklist-icon acceptance-icon"><i class="fa-solid fa-check"></i></div><div class="quote-text"><p><strong>Решение:</strong>\1</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Ошибка:
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<p>\s*<strong>Ошибка(?:\s*[-—:]\s*|\s+)(.*?)</strong>(.*?)</p>\s*</blockquote>',
        r'<div class="quote-block danger-block"><div class="quote-content"><div class="checklist-icon danger-icon"><i class="fa-solid fa-xmark"></i></div><div class="quote-text"><p><strong>Ошибка \1 </strong>\2</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<strong>Ошибка(?:\s*[-—:]\s*|\s+)(.*?)</strong>(.*?)\s*</blockquote>',
        r'<div class="quote-block danger-block"><div class="quote-content"><div class="checklist-icon danger-icon"><i class="fa-solid fa-xmark"></i></div><div class="quote-text"><p><strong>Ошибка \1 </strong>\2</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )

    # ВАЖНО:
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<p>\s*<strong>ВАЖНО:?</strong>(.*?)</p>\s*</blockquote>',
        r'<div class="quote-block warn-block"><div class="quote-content"><div class="checklist-icon warn-icon"><i class="fa-solid fa-exclamation"></i></div><div class="quote-text"><p><strong>ВАЖНО:</strong>\1</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    html_body = re.sub(
        r'<blockquote[^>]*>\s*<strong>ВАЖНО:?</strong>(.*?)\s*</blockquote>',
        r'<div class="quote-block warn-block"><div class="quote-content"><div class="checklist-icon warn-icon"><i class="fa-solid fa-exclamation"></i></div><div class="quote-text"><p><strong>ВАЖНО:</strong>\1</p></div></div></div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not re.search(r'<h1[^>]*>', html_body, re.IGNORECASE):
        html_body = f'<h1>{meta_title}</h1>\n\n' + html_body

    subtitle = meta.get("subtitle") or meta_description or ""
    if subtitle:
        lead_html = f'\n<p class="lead-text">{subtitle}</p>\n'
        html_body = re.sub(
            r'(<h1[^>]*>.*?</h1>)',
            rf'\1{lead_html}',
            html_body,
            count=1,
            flags=re.IGNORECASE
        )

    char_count = len(article_content)
    reading_time = max(1, char_count // 1500)
    char_count_formatted = f"{char_count:,}".replace(",", " ") + " симв."
    reading_time_formatted = f"~{reading_time} мин."
    
    # HTML Шаблон (Премиум B2B стиль с сайдбаром)
    html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{meta_description}">
    <meta name="keywords" content="{keywords_list_meta}">
    <title>{meta_title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:ital,wght@0,600;0,700;0,800;0,900;1,600&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        :root {
            --bg-color: #fdfdfc;
            --surface-color: #ffffff;
            --text-main: #1c1917;
            --text-muted: #57534e;
            --primary: #78350f;
            --primary-light: #fef3c7;
            --accent: #b45309;
            --border: #e7e5e4;
            --shadow: 0 4px 20px -2px rgb(120 53 15 / 0.03);
            --shadow-lg: 0 10px 30px -5px rgb(120 53 15 / 0.05);
        }

        .highlight-box {
            padding: 20px 24px;
            border-radius: 8px;
            margin: 28px 0;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            transition: all 0.2s ease;
        }

        .highlight-box:hover {
            transform: translateX(2px);
        }

        .highlight-box.error-box {
            border-left: 5px solid #dc2626;
            background-color: #fff5f5;
            color: #7f1d1d;
            font-style: italic;
        }

        .highlight-box.error-box strong {
            font-weight: 700;
            color: #b91c1c;
            font-style: normal;
            margin-right: 4px;
        }

        .highlight-box.warn-box, .highlight-box.important {
            border-left: 5px solid #d97706;
            background-color: #fffbeb;
            color: #78350f;
            font-style: italic;
        }

        .highlight-box.warn-box strong, .highlight-box.important strong {
            font-weight: 700;
            color: #b45309;
            font-style: normal;
            margin-right: 4px;
        }

        .quote-block {
            margin: 24px 0;
            padding: 20px 24px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
            border: 1px solid transparent;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .quote-block:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        }

        .quote-content {
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }

        .checklist-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            font-size: 1.1rem;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .quote-text {
            font-size: 1rem;
            line-height: 1.6;
            color: var(--text-main);
        }

        .quote-text p {
            margin: 0;
            text-align: left;
        }

        .acceptance-block {
            background-color: #f0fdf4;
            border-color: #bbf7d0;
        }
        
        .acceptance-icon {
            background-color: #dcfce7;
            color: #16a34a;
            border: 1px solid #bbf7d0;
        }

        .danger-block {
            background-color: #fef2f2;
            border-color: #fecaca;
        }

        .danger-icon {
            background-color: #fee2e2;
            color: #dc2626;
            border: 1px solid #fecaca;
        }

        .warn-block {
            background-color: #fffbeb;
            border-color: #fef3c7;
        }

        .warn-icon {
            background-color: #fef3c7;
            color: #d97706;
            border: 1px solid #fde68a;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Lora', Georgia, serif;
            margin: 0;
            padding: 0;
            line-height: 1.8;
        }

        .article-card > p:first-of-type::first-letter {
            font-family: 'Playfair Display', serif;
            font-size: 3.2rem;
            font-weight: 900;
            float: left;
            line-height: 0.9;
            margin-right: 12px;
            margin-top: 6px;
            color: var(--primary);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
            box-sizing: border-box;
        }

        header {
            margin-bottom: 24px;
            text-align: left;
        }

        .badge {
            display: inline-block;
            padding: 6px 16px;
            background-color: var(--primary-light);
            color: var(--primary);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-radius: 9999px;
            margin-bottom: 16px;
            border: 1px solid rgba(16, 185, 129, 0.15);
        }

        .layout {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 40px;
        }

        @media (max-width: 1024px) {
            .layout {
                grid-template-columns: 1fr;
            }
        }

        .article-card {
            background-color: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 50px;
            box-shadow: var(--shadow-lg);
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .sticky-sidebar {
            position: sticky;
            top: 40px;
        }

        .card {
            background-color: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }

        .card-title {
            font-family: 'Playfair Display', serif;
            font-size: 0.875rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }

        h1, h2, h3, h4 {
            font-family: 'Playfair Display', Georgia, serif;
            color: var(--text-main);
            font-weight: 700;
        }

        h1 {
            font-size: 2.5rem;
            line-height: 1.25;
            margin-top: 0;
            margin-bottom: 24px;
        }

        .lead-text {
            font-size: 1.25rem;
            line-height: 1.6;
            color: var(--text-muted);
            margin-bottom: 32px;
            font-weight: 400;
            border-left: 3px solid var(--primary);
            padding-left: 16px;
        }

        h2 {
            font-size: 1.65rem;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        h3 {
            font-size: 1.25rem;
            margin-top: 32px;
            margin-bottom: 16px;
        }

        p {
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.05rem;
            color: var(--text-main);
            text-align: justify;
        }

        ul, ol {
            margin-top: 0;
            margin-bottom: 24px;
            padding-left: 24px;
        }

        li {
            margin-bottom: 10px;
            font-size: 1.05rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 32px 0;
            font-size: 0.95rem;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }

        th, td {
            padding: 14px 20px;
            text-align: left;
        }

        th {
            background-color: #f1f5f9;
            color: var(--text-main);
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }

        td {
            border-bottom: 1px solid var(--border);
            background-color: var(--surface-color);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background-color: #f8fafc;
        }

        blockquote {
            border-left: 4px solid var(--primary);
            padding: 16px 24px;
            margin: 32px 0;
            background-color: #f8fafc;
            border-radius: 0 12px 12px 0;
            font-style: italic;
            color: var(--text-muted);
        }

        .highlight-box {
            padding: 20px 24px;
            border-radius: 8px;
            margin: 28px 0;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            transition: all 0.2s ease;
        }

        .highlight-box:hover {
            transform: translateX(2px);
        }

        .highlight-box.error-box {
            border-left: 5px solid #dc2626;
            background-color: #fff5f5;
            color: #7f1d1d;
            font-style: italic;
        }

        .highlight-box.error-box strong {
            font-weight: 700;
            color: #b91c1c;
            font-style: normal;
            margin-right: 4px;
        }

        .highlight-box.warn-box, .highlight-box.important {
            border-left: 5px solid #d97706;
            background-color: #fffbeb;
            color: #78350f;
            font-style: italic;
        }

        .highlight-box.warn-box strong, .highlight-box.important strong {
            font-weight: 700;
            color: #b45309;
            font-style: normal;
            margin-right: 4px;
        }

        .meta-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px dashed var(--border);
            font-size: 0.875rem;
        }

        .meta-item:last-child {
            border-bottom: none;
        }

        .meta-label {
            color: var(--text-muted);
            font-weight: 500;
        }

        .meta-value {
            font-weight: 600;
            color: var(--text-main);
        }

        .keyword-tag {
            display: inline-block;
            padding: 4px 10px;
            background-color: #f1f5f9;
            color: var(--text-muted);
            font-size: 0.75rem;
            font-weight: 500;
            border-radius: 6px;
            margin-right: 6px;
            margin-bottom: 8px;
            border: 1px solid var(--border);
        }

        .cover-image {
            width: 100%;
            height: auto;
            max-height: 480px;
            object-fit: cover;
            border-radius: 16px;
            margin: 0 0 40px 0;
            box-shadow: var(--shadow-lg);
            display: block;
        }

        .section-image {
            width: 100%;
            height: auto;
            max-height: 250px;
            object-fit: cover;
            border-radius: 12px;
            margin: 40px 0;
            box-shadow: var(--shadow);
            display: block;
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin: 24px 0;
            box-shadow: var(--shadow);
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge">{direction} • {type_name}</span>
        </header>

        <div class="layout">
            <div class="article-card">
                {html_body}
            </div>

            <div class="sidebar">
                <div class="sticky-sidebar">
                    <div class="card">
                        <h4 class="card-title">Паспорт статьи</h4>
                        <div class="meta-item">
                            <span class="meta-label">Тип материала</span>
                            <span class="meta-value">{type_name}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Направление</span>
                            <span class="meta-value">{direction}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Длина статьи</span>
                            <span class="meta-value">{char_count_formatted}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Время чтения</span>
                            <span class="meta-value">{reading_time_formatted}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Создано нейросетью</span>
                            <span class="meta-value">{date}</span>
                        </div>
                    </div>

                    <div class="card">
                        <h4 class="card-title">Оптимизация (SEO/GEO)</h4>
                        <div style="margin-bottom: 12px;">
                            <span class="meta-label" style="font-size: 0.75rem; display: block; margin-bottom: 4px;">META TITLE</span>
                            <span style="font-size: 0.875rem; font-weight: 600; display: block; color: var(--text-main);">{meta_title}</span>
                        </div>
                        <div style="margin-bottom: 16px;">
                            <span class="meta-label" style="font-size: 0.75rem; display: block; margin-bottom: 4px;">META DESCRIPTION</span>
                            <span style="font-size: 0.85rem; color: var(--text-muted); display: block; line-height: 1.4; text-align: justify;">{meta_description}</span>
                        </div>
                        <div>
                            <span class="meta-label" style="font-size: 0.75rem; display: block; margin-bottom: 8px;">КЛЮЧЕВЫЕ СЛОВА</span>
                            <div>
                                {keywords_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    # Безопасная замена вместо .format(), чтобы избежать KeyError на CSS стилях
    direction_val = direction or "Бизнес"
    html_content = html_template.replace("{meta_title}", meta_title) \
                                 .replace("{direction}", direction_val) \
                                 .replace("{type_name}", type_name) \
                                 .replace("{html_body}", html_body) \
                                 .replace("{char_count_formatted}", char_count_formatted) \
                                 .replace("{reading_time_formatted}", reading_time_formatted) \
                                 .replace("{date}", datetime.now().strftime("%Y-%m-%d %H:%M")) \
                                 .replace("{meta_description}", meta_description) \
                                 .replace("{keywords_list_meta}", ", ".join(keywords_list)) \
                                 .replace("{keywords_html}", keywords_html)

    html_path = output_dir / "article.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"🌐 HTML превью: {html_path}")


def save_result(state, output_dir: Path):
    """Сохранить результат pipeline в файлы."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Статья в Markdown
    article_path = output_dir / "article.md"
    article_content = state.final_article or state.draft or "[Статья не сгенерирована]"

    # Добавляем мета-информацию в начало
    meta = state.final_meta or {}
    frontmatter = "---\n"
    frontmatter += f"title: \"{meta.get('title', state.topic)}\"\n"
    frontmatter += f"description: \"{meta.get('description', '')}\"\n"
    frontmatter += f"keywords: {json.dumps(meta.get('keywords', []), ensure_ascii=False)}\n"
    frontmatter += f"type: {state.article_type}\n"
    frontmatter += f"direction: {state.direction}\n"
    frontmatter += f"generated: {datetime.now().isoformat()}\n"
    frontmatter += f"sheriff_iterations: {state.sheriff_iterations}\n"
    frontmatter += f"mirror_iterations: {state.mirror_iterations}\n"
    frontmatter += "---\n\n"

    article_path.write_text(frontmatter + article_content, encoding="utf-8")
    logger.info(f"📄 Статья: {article_path}")

    # 2. SEO-пакет
    if state.seo_package:
        seo_path = output_dir / "seo_package.json"
        seo_path.write_text(
            json.dumps(state.seo_package, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"🚀 SEO: {seo_path}")

    # 3. Промпты для изображений
    if state.image_prompts:
        img_path = output_dir / "image_prompts.json"
        img_path.write_text(
            json.dumps(state.image_prompts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"🎨 Изображения: {img_path}")

    # 4. Полный лог pipeline (для отладки)
    debug_path = output_dir / "pipeline_debug.json"
    debug_data = {
        "topic": state.topic,
        "article_type": state.article_type,
        "direction": state.direction,
        "status": state.status,
        "error": state.error,
        "steps_completed": state.steps_completed,
        "sheriff_iterations": state.sheriff_iterations,
        "mirror_iterations": state.mirror_iterations,
        "brain_output": state.brain_output,
        "facts_summary": {
            "count": len(state.facts.get("facts", [])) if isinstance(state.facts, dict) else 0,
        },
        "sheriff_review": state.sheriff_review,
        "mirror_review": state.mirror_review,
        "tokens": {
            "total": getattr(state, 'total_tokens', 0),
            "prompt": getattr(state, 'total_prompt_tokens', 0),
            "completion": getattr(state, 'total_completion_tokens', 0),
            "by_agent": getattr(state, 'tokens_by_agent', {}),
        },
    }
    debug_path.write_text(
        json.dumps(debug_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"🔍 Debug: {debug_path}")

    # 5. Schema.org (если есть)
    schema = state.seo_package.get("schema_json_ld") if state.seo_package else None
    if schema:
        schema_path = output_dir / "schema.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"📋 Schema.org: {schema_path}")

    # 6. HTML Превью для демонстрации верстки (НЕЙРОЦЕХ)
    try:
        save_html_preview(state, output_dir)
    except Exception as html_err:
        logger.error(f"❌ Ошибка генерации HTML-превью: {html_err}")

    # 7. Паспорт статьи (passport.txt)
    char_count = len(article_content)
    reading_time = max(1, char_count // 1500)
    passport_lines = [
        "============================================================",
        "                  ПАСПОРТ СТАТЬИ / REPORT",
        "============================================================",
        f"Тема: {state.topic}",
        f"Тип статьи: {state.article_type}",
        f"Направление (дирекция): {state.direction or 'авто'}",
        f"Стиль (ID): {getattr(state, 'style_id', '') or 'авто'}",
        f"Целевой объем: {getattr(state, 'custom_chars', 0)} символов",
        f"Фактический объем: {char_count} символов",
        f"Приблизительное время чтения: {reading_time} мин.",
        f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "------------------------------------------------------------",
        "МЕТРИКИ РАБОТЫ АГЕНТОВ:",
        f"Итераций Sheriff (корректор): {state.sheriff_iterations}",
        f"Итераций Mirror (редактор): {state.mirror_iterations}",
        "------------------------------------------------------------",
        "РАСХОД ТОКЕНОВ (TOKEN METRICS):",
        f"Всего токенов: {getattr(state, 'total_tokens', 0):,}",
        f"Промпт токены: {getattr(state, 'total_prompt_tokens', 0):,}",
        f"Выходные токены: {getattr(state, 'total_completion_tokens', 0):,}",
    ]
    
    tokens_by_agent = getattr(state, 'tokens_by_agent', {})
    if tokens_by_agent:
        passport_lines.append("Детализация по агентам:")
        for aid, d in tokens_by_agent.items():
            passport_lines.append(f"  - {aid}: {d['prompt'] + d['completion']:,} токенов ({d['calls']} вызовов)")
            
    passport_lines.append("============================================================")
    passport_path = output_dir / "passport.txt"
    try:
        passport_path.write_text("\n".join(passport_lines), encoding="utf-8")
        logger.info(f"📋 Паспорт статьи сохранен в: {passport_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения паспорта статьи: {e}")

    # 8. SEO-метаданные (seo.txt)
    meta_title = meta.get("title") or state.topic
    meta_description = meta.get("description") or "Статья сгенерирована мультиагентной системой Copywriter."
    keywords_list = meta.get("keywords") or []
    if isinstance(keywords_list, str):
        keywords_list = [k.strip() for k in keywords_list.split(",") if k.strip()]
        
    seo_lines = [
        "============================================================",
        "                SEO & OPTIMIZATION METADATA",
        "============================================================",
        f"META TITLE: {meta_title}",
        "------------------------------------------------------------",
        f"META DESCRIPTION: {meta_description}",
        "------------------------------------------------------------",
        "KEYWORDS (КЛЮЧЕВЫЕ СЛОВА):",
    ]
    for i, kw in enumerate(keywords_list, 1):
        seo_lines.append(f"  {i}. {kw}")
        
    if schema:
        seo_lines.append("------------------------------------------------------------")
        seo_lines.append("SCHEMA.ORG JSON-LD:")
        seo_lines.append(json.dumps(schema, ensure_ascii=False, indent=2))
        
    seo_lines.append("============================================================")
    seo_path = output_dir / "seo.txt"
    try:
        seo_path.write_text("\n".join(seo_lines), encoding="utf-8")
        logger.info(f"🚀 SEO-метаданные сохранены в: {seo_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения SEO-метаданных: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🖊️ Генератор статей — мультиагентный pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python generate.py "Как открыть ООО" --type seo --dir юридическое
  python generate.py "10 ошибок УСН" --type seo --style checklist --dir налоги --no-scout
  python generate.py "Разбор налоговой реформы 2026" --type longread --dir налоги
        """,
    )

    parser.add_argument("topic", help="Тема статьи")
    parser.add_argument(
        "--type", "-t",
        dest="article_type",
        default="seo",
        choices=["seo", "longread"],
        help="Тип статьи по размеру: seo (до 10000) / longread (до 30000). default: seo",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="direction",
        default="",
        help="Направление: налоги / юридическое / бизнес / финансы / экономика",
    )
    parser.add_argument("--no-scout", action="store_true", help="Пропустить Scout (нет SearXNG)")
    parser.add_argument("--images", action="store_true", help="Включить генерацию картинок (Artist)")
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Папка для результатов (default: output/)",
    )
    parser.add_argument(
        "--model-override",
        default=None,
        help="Переопределить модель для всех агентов (например: gpt-4o)",
    )
    parser.add_argument(
        "--style", "-s",
        dest="style_id",
        default="",
        help="ID стиля: checklist / analysis / reference / law_review / case_study",
    )
    parser.add_argument(
        "--chars", "-c",
        dest="custom_chars",
        type=int,
        default=0,
        help="Целевой объём статьи в символах (0 = по умолчанию из стиля)",
    )
    parser.add_argument(
        "--description", "--desc",
        dest="description",
        default="",
        help="Описание статьи, контекст, ЦА, факты",
    )
    parser.add_argument(
        "--style-nuances", "--nuances",
        dest="style_nuances",
        default="",
        help="Нюансы стиля и ручные требования к тональности",
    )
    parser.add_argument(
        "--instructions", "--inst",
        dest="additional_instructions",
        default="",
        help="Дополнительные инструкции по написанию статьи",
    )
    parser.add_argument(
        "--keywords", "--keys",
        dest="keywords",
        default="",
        help="Ключевые слова через запятую",
    )
    parser.add_argument(
        "--provider", "-p",
        dest="provider",
        default="deepseek",
        choices=["deepseek", "kie", "openai"],
        help="Провайдер LLM для генерации текста (default: deepseek)",
    )
    parser.add_argument(
        "--model", "-m",
        dest="model",
        default="",
        help="Конкретная модель для генерации (например: deepseek-v4-pro, claude-4.7, gpt-5.5)",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        dest="quality_mode",
        default=None,
        help="Включить режим супер-качества QUALITY_MODE",
    )

    args = parser.parse_args()

    # Проверка API ключа
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY не найден в .env!")
        sys.exit(1)

    # Переопределение модели (если указано)
    if args.model_override:
        from agents.registry import AGENTS
        for agent in AGENTS.values():
            agent.model = args.model_override
        logger.info(f"⚙️ Модель переопределена: {args.model_override}")
    else:
        logger.info(f"⚙️ Запуск с провайдером: {args.provider}, модель: {args.model or 'авто'}")

    # Qdrant клиент
    qdrant_client = None
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        qdrant_client = QdrantClient(
            host=host,
            port=port,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            https=False
        )
        # Проверка соединения
        qdrant_client.get_collections()
        logger.info(f"✅ Qdrant: {host}:{port}")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant недоступен: {e}. RAG будет отключен.")
        qdrant_client = None

    # Pipeline
    from agents.pipeline import Pipeline
    pipe = Pipeline(
        openai_api_key=api_key,
        qdrant_client=qdrant_client,
    )

    # Подтягиваем авто-объем символов из patterns.py, если не указан
    from agents.patterns import PATTERNS
    if not args.custom_chars and args.article_type in PATTERNS:
        args.custom_chars = PATTERNS[args.article_type].get("target_chars", 8000)

    logger.info(f"\n{'='*60}")
    logger.info(f"📝 Тема: {args.topic}")
    logger.info(f"📋 Тип: {args.article_type}")
    logger.info(f"🧭 Направление: {args.direction or 'авто'}")
    logger.info(f"🔌 Провайдер: {args.provider}")
    logger.info(f"🤖 Модель: {args.model or 'авто'}")
    logger.info(f"📝 Описание: {args.description[:60] + '...' if args.description else 'нет'}")
    logger.info(f"🎨 Стиль: {args.style_id or 'авто (по типу)'}")
    logger.info(f"📏 Объём: ~{args.custom_chars} символов (авто)" if args.custom_chars else f"📏 Объём: авто")
    logger.info(f"📡 Scout: {'❌ пропущен' if args.no_scout else '✅'}")
    logger.info(f"🎨 Artist: {'✅' if args.images else '❌ пропущен'}")
    logger.info(f"{'='*60}\n")

    # Сохранение (определяем путь заранее, чтобы передать в пайплайн для картинок)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    slug = slugify(args.topic)
    output_dir = Path(args.output) / f"{timestamp}_{slug}"

    # Парсинг ключевых слов и размера для пайплайна
    keywords_list = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None
    size_str = "long" if args.article_type == "longread" else "short"

    # Запуск
    state = pipe.run(
        topic=args.topic,
        article_type=args.article_type,
        direction=args.direction,
        skip_scout=args.no_scout,
        skip_images=not args.images,
        style_id=args.style_id,
        custom_chars=args.custom_chars,
        output_dir=str(output_dir),
        provider=args.provider,
        model=args.model,
        description=args.description,
        style_nuances=args.style_nuances,
        additional_instructions=args.additional_instructions,
        quality_mode=args.quality_mode,
        keywords=keywords_list,
        size=size_str,
    )

    save_result(state, output_dir)

    # Итог
    print(f"\n{'='*60}")
    if state.status == "completed":
        print(f"✅ Статья готова: {output_dir / 'article.md'}")
        print(f"   Sheriff итераций: {state.sheriff_iterations}")
        print(f"   Mirror итераций: {state.mirror_iterations}")
        total_tokens = getattr(state, 'total_tokens', 0)
        if total_tokens:
            print(f"   📊 Токенов всего: {total_tokens:,}")
            tokens_by_agent = getattr(state, 'tokens_by_agent', {})
            if tokens_by_agent:
                print(f"   {'─' * 30}")
                for aid, d in tokens_by_agent.items():
                    print(f"   {aid:>15}: {d['prompt']+d['completion']:>8,} ({d['calls']} вызовов)")
    elif state.status == "budget_exhausted":
        print(f"💸 Баланс исчерпан. Частичный результат: {output_dir}")
        print(f"   Шаги завершены: {', '.join(state.steps_completed)}")
    else:
        print(f"❌ Ошибка: {state.error}")
        print(f"   Шаги завершены: {', '.join(state.steps_completed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
