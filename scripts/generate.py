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


def save_html_preview(state, output_dir: Path):
    """Сгенерировать красивый, адаптивный HTML-просмотрщик статьи (дизайн 'НЕЙРОЦЕХ / Журнальный разворот')."""
    article_content = state.final_article or state.draft or ""
    if not article_content:
        return

    # Нормализуем переводы строк для совместимости Windows/Linux и надежной работы парсера
    article_content = article_content.replace("\r\n", "\n")

    # Определение человеческого типа статьи
    type_names = {
        "seo": "СЕО-статья",
        "longread": "Лонгрид",
    }
    # Если выбран конкретный стиль-пресет — показываем его человеческое имя,
    # иначе человеческое имя размерного типа (seo/longread).
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
        # Пытаемся спасти сначала из сырого ответа Booster, так как при поломке JSON данные там
        raw_booster = state.seo_package.get("raw_response", "") if state.seo_package else ""
        extracted = _extract_meta_from_text(raw_booster) if raw_booster else {}
        
        # Если там нет, то ищем в самой статье
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

    # Сгенерировать теги ключевых слов
    keywords_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords_list])
    if not keywords_html:
        keywords_html = '<span class="keyword-tag">налоги</span><span class="keyword-tag">бизнес</span>'
        
    # Преобразуем Markdown в HTML
    html_body = ""
    try:
        import markdown
        # Использование официального парсера с таблицами
        html_body = markdown.markdown(article_content, extensions=['tables', 'fenced_code', 'nl2br'])
    except ImportError:
        # Простой встроенный регулярный парсер (чтобы работало без пипа)
        html = article_content
        
        # Экранирование базовых тегов
        html = html.replace("<", "&lt;").replace(">", "&gt;")
        
        # Заголовки
        html = re.sub(r'^#\s+(.*?)$', r'<h1 class="text-3xl font-bold my-6">\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^##\s+(.*?)$', r'<h2 class="text-2xl font-bold my-4 border-b pb-2">\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^###\s+(.*?)$', r'<h3 class="text-xl font-bold my-3">\1</h3>', html, flags=re.MULTILINE)
        
        # Картинки в Markdown: ![caption](url)
        html = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1" />', html)
        
        # Жирный и курсив
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Цитаты
        html = re.sub(r'^&gt;\s+(.*?)$', r'<blockquote class="border-l-4 border-primary pl-4 my-4 italic text-textMuted">\1</blockquote>', html, flags=re.MULTILINE)
        
        # Списки
        html = re.sub(r'^\s*-\s+(.*?)$', r'<li class="list-disc ml-4">\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\s*\*\s+(.*?)$', r'<li class="list-disc ml-4">\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\s*\d+\.\s+(.*?)$', r'<li class="list-decimal ml-4">\1</li>', html, flags=re.MULTILINE)
        
        # Разделение по абзацам с нормализацией пустых строк с пробелами/табами
        html = re.sub(r'\n\s*\n', '\n\n', html)
        paragraphs = html.split('\n\n')
        for i, p in enumerate(paragraphs):
            p_strip = p.strip()
            if not p_strip:
                continue
            if not p_strip.startswith('<h') and not p_strip.startswith('<blockquote') and not p_strip.startswith('<ul') and not p_strip.startswith('<ol') and not p_strip.startswith('<li') and not p_strip.startswith('<img') and not p_strip.startswith('<table'):
                paragraphs[i] = f'<p class="mb-4 text-justify leading-relaxed">{p_strip}</p>'
        html_body = "\n\n".join(paragraphs)
        
    # Назначаем классы изображениям для красивой стилизации (Обложка сверху, тонкие разделители в тексте)
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

    # Кастомная подсветка блоков ВАЖНО и Ошибка с премиальными классами и эмодзи
    html_body = re.sub(
        r'<blockquote>\s*<p>\s*<strong>Ошибка\s*—\s*</strong>(.*?)</p>\s*</blockquote>',
        r'<div class="highlight-box error-box"><strong>❌ Ошибка — </strong>\1</div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    html_body = re.sub(
        r'<blockquote>\s*<strong>Ошибка\s*—\s*</strong>(.*?)\s*</blockquote>',
        r'<div class="highlight-box error-box"><strong>❌ Ошибка — </strong>\1</div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    html_body = re.sub(
        r'<blockquote>\s*<p>\s*<strong>ВАЖНО:?</strong>(.*?)</p>\s*</blockquote>',
        r'<div class="highlight-box warn-box"><strong>⚠️ ВАЖНО:</strong>\1</div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    html_body = re.sub(
        r'<blockquote>\s*<strong>ВАЖНО:?</strong>(.*?)\s*</blockquote>',
        r'<div class="highlight-box warn-box"><strong>⚠️ ВАЖНО:</strong>\1</div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Кастомная подсветка блоков ПОСЛЕСЛОВИЕ / ИТОГ
    html_body = re.sub(
        r'<p>\s*(Послесловие|Итог):?\s*(.*?)</p>',
        r'<div class="highlight-box important"><strong>🎯 ИТОГ:</strong> \2</div>',
        html_body,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Если в html_body нет заголовка h1, вставляем красивый заголовок первого уровня
    if not re.search(r'<h1[^>]*>', html_body, re.IGNORECASE):
        html_body = f'<h1>{meta_title}</h1>\n\n' + html_body

    # Вычисляем примерное время чтения (1500 символов в минуту)
    char_count = len(article_content)
    reading_time = max(1, char_count // 1500)
    
    # HTML Шаблон
    html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{meta_title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --primary: #10b981;
            --primary-light: #ecfdf5;
            --accent: #3b82f6;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.7;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
            box-sizing: border-box;
        }}

        header {{
            margin-bottom: 24px;
            text-align: left;
        }}

        .badge {{
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
        }}

        .layout {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 40px;
        }}

        @media (max-width: 1024px) {{
            .layout {{
                grid-template-columns: 1fr;
            }}
        }}

        .article-card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 50px;
            box-shadow: var(--shadow-lg);
        }}

        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .sticky-sidebar {{
            position: sticky;
            top: 40px;
        }}

        .card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }}

        .card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.875rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}

        /* Typography */
        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
            font-weight: 700;
        }}

        h1 {{
            font-size: 2.5rem;
            line-height: 1.25;
            margin-top: 0;
            margin-bottom: 24px;
        }}

        h2 {{
            font-size: 1.65rem;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}

        h3 {{
            font-size: 1.25rem;
            margin-top: 32px;
            margin-bottom: 16px;
        }}

        p {{
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.05rem;
            color: var(--text-main);
            text-align: justify;
        }}

        /* Lists */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 24px;
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 10px;
            font-size: 1.05rem;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 32px 0;
            font-size: 0.95rem;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }}

        th, td {{
            padding: 14px 20px;
            text-align: left;
        }}

        th {{
            background-color: #f1f5f9;
            color: var(--text-main);
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }}

        td {{
            border-bottom: 1px solid var(--border);
            background-color: var(--surface-color);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: #f8fafc;
        }}

        /* Custom highlights */
        blockquote {{
            border-left: 4px solid var(--primary);
            padding: 16px 24px;
            margin: 32px 0;
            background-color: #f8fafc;
            border-radius: 0 12px 12px 0;
            font-style: italic;
            color: var(--text-muted);
        }}

        .highlight-box {{
            padding: 20px 24px;
            border-radius: 8px;
            margin: 28px 0;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            transition: all 0.2s ease;
        }}

        .highlight-box:hover {{
            transform: translateX(2px);
        }}

        .highlight-box.error-box {{
            border-left: 5px solid #dc2626;
            background-color: #fff5f5;
            color: #7f1d1d;
            font-style: italic;
        }}

        .highlight-box.error-box strong {{
            font-weight: 700;
            color: #b91c1c;
            font-style: normal;
            margin-right: 4px;
        }}

        .highlight-box.warn-box, .highlight-box.important {{
            border-left: 5px solid #d97706;
            background-color: #fffbeb;
            color: #78350f;
            font-style: italic;
        }}

        .highlight-box.warn-box strong, .highlight-box.important strong {{
            font-weight: 700;
            color: #b45309;
            font-style: normal;
            margin-right: 4px;
        }}

        .meta-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px dashed var(--border);
            font-size: 0.875rem;
        }}

        .meta-item:last-child {{
            border-bottom: none;
        }}

        .meta-label {{
            color: var(--text-muted);
            font-weight: 500;
        }}

        .meta-value {{
            font-weight: 600;
            color: var(--text-main);
        }}

        .keyword-tag {{
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
        }}

        .cover-image {{
            width: 100%;
            height: auto;
            max-height: 480px;
            object-fit: cover;
            border-radius: 16px;
            margin: 0 0 40px 0;
            box-shadow: var(--shadow-lg);
            display: block;
        }}

        .section-image {{
            width: 100%;
            height: auto;
            max-height: 250px;
            object-fit: cover;
            border-radius: 12px;
            margin: 40px 0;
            box-shadow: var(--shadow);
            display: block;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin: 24px 0;
            box-shadow: var(--shadow);
            display: block;
        }}
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
                            <span class="meta-value">{char_count:,} симв.</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Время чтения</span>
                            <span class="meta-value">~{reading_time} мин.</span>
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

    html_content = html_template.format(
        meta_title=meta_title,
        direction=direction,
        type_name=type_name,
        html_body=html_body,
        char_count=char_count,
        reading_time=reading_time,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        meta_description=meta_description,
        keywords_html=keywords_html
    )

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
    parser.add_argument("--no-images", action="store_true", help="Пропустить Artist")
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
    logger.info(f"🎨 Artist: {'❌ пропущен' if args.no_images else '✅'}")
    logger.info(f"{'='*60}\n")

    # Сохранение (определяем путь заранее, чтобы передать в пайплайн для картинок)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    slug = slugify(args.topic)
    output_dir = Path(args.output) / f"{timestamp}_{slug}"

    # Запуск
    state = pipe.run(
        topic=args.topic,
        article_type=args.article_type,
        direction=args.direction,
        skip_scout=args.no_scout,
        skip_images=args.no_images,
        style_id=args.style_id,
        custom_chars=args.custom_chars,
        output_dir=str(output_dir),
        provider=args.provider,
        model=args.model,
        description=args.description,
        style_nuances=args.style_nuances,
        additional_instructions=args.additional_instructions,
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
