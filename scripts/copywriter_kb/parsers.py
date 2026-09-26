"""
Универсальный парсер файлов для Knowledge Base Pipeline.

Поддержка: PDF, FB2, DOCX, ODT, TXT
FB2 — приоритетный формат (парсинг по XML-тегам <p>, <title>).
"""
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kb.parsers")


def extract_text(file_path: Path, source_type: Optional[str] = None) -> Optional[str]:
    """
    Универсальная точка входа: определяет формат и вызывает нужный парсер.

    source_type: "law" — юридический текст (не срезаем одиночные числа-строки:
                 в законах это номера частей/пунктов, а не номера страниц).

    Returns:
        Чистый текст или None при ошибке
    """
    suffix = file_path.suffix.lower()

    parsers = {
        ".pdf": _parse_pdf,
        ".fb2": _parse_fb2,
        ".txt": _parse_txt,
        ".docx": _parse_docx,
        ".odt": _parse_odt,
    }

    parser = parsers.get(suffix)
    if not parser:
        logger.warning(f"⚠️ Неподдерживаемый формат: {suffix} ({file_path.name})")
        return None

    try:
        text = parser(file_path)
        if text:
            text = _clean_text(text, is_law=(source_type == "law"))
        return text if text and len(text) > 50 else None
    except Exception as e:
        logger.error(f"❌ Ошибка парсинга {file_path.name}: {e}")
        return None


# ============================================================
# PDF — PyMuPDF (fitz)
# ============================================================

def _parse_pdf(file_path: Path) -> str:
    """Извлечь текст из PDF через PyMuPDF"""
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    doc.close()

    return text


# ============================================================
# FB2 — XML парсинг (приоритетный формат)
# ============================================================

def _parse_fb2(file_path: Path) -> str:
    """
    Извлечь текст из FB2 по XML-тегам <p>, <title>, <v>, <subtitle>.
    FB2 — приоритетный формат для сохранения семантической структуры абзацев.

    Особенности реальных FB2:
    - именованные HTML-entities (&nbsp;, &mdash;), которые XML не знает;
    - чтение ТОЛЬКО из <body> (раньше аннотация из <description> попадала в текст);
    - битые файлы: фолбэк на lxml recover, затем на грубый regex.
    """
    import html.entities
    import xml.etree.ElementTree as ET

    raw_bytes = file_path.read_bytes()
    content = _decode_bytes(raw_bytes)

    # 1. Именованные entity, кроме XML-предопределённых, заменяем на символы
    def _ent(m: "re.Match") -> str:
        name = m.group(1)
        return html.entities.html5.get(name + ";", m.group(0))

    content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)([a-zA-Z][a-zA-Z0-9]*);', _ent, content)

    root = None
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        try:
            from lxml import etree as _lxml
            parser = _lxml.XMLParser(recover=True, huge_tree=True)
            root = _lxml.fromstring(content.encode("utf-8", "replace"), parser=parser)
        except ImportError:
            root = None

    if root is not None:
        parts: list[str] = []
        # Только тела документа: аннотация/издательские данные не должны попадать в текст
        bodies = [el for el in root.iter() if _strip_ns(el.tag) == "body"]
        scan = bodies if bodies else [root]
        for el_root in scan:
            for el in el_root.iter():
                tag = _strip_ns(el.tag)
                if tag == "title":
                    title_text = _get_all_text(el)
                    if title_text.strip():
                        parts.append(f"\n## {title_text.strip()}\n")
                elif tag == "p":
                    p_text = _get_all_text(el)
                    if p_text.strip():
                        parts.append(p_text.strip())
                elif tag == "v":
                    v_text = _get_all_text(el)
                    if v_text.strip():
                        parts.append(v_text.strip())
                elif tag == "subtitle":
                    s_text = _get_all_text(el)
                    if s_text.strip():
                        parts.append(f"\n## {s_text.strip()}\n")
                elif tag == "empty-line":
                    parts.append("")
        text = "\n".join(parts)
        if len(text) > 200:
            return text

    # 2. Грубый фолбэк: вырезать теги регуляркой
    text = re.sub(r"<[^>]+>", " ", content)
    text = html_re_unescape(text)
    return re.sub(r"[ \t]{2,}", " ", text)


def html_re_unescape(text: str) -> str:
    import html
    return html.unescape(text)


def _strip_ns(tag: str) -> str:
    """Убирает namespace из XML-тега: {http://...}p → p"""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_all_text(element) -> str:
    """Рекурсивно извлекает весь текст из XML-элемента, включая вложенные теги"""
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.append(_get_all_text(child))
        if child.tail:
            texts.append(child.tail)
    return " ".join(texts)


# ============================================================
# TXT — Plain text с автодетекцией кодировки
# ============================================================

def _parse_txt(file_path: Path) -> str:
    """Прочитать TXT с автоматической детекцией кодировки"""
    raw_bytes = file_path.read_bytes()
    return _decode_bytes(raw_bytes)


# ============================================================
# DOCX — python-docx
# ============================================================

def _parse_docx(file_path: Path) -> str:
    """Извлечь текст из DOCX"""
    from docx import Document

    doc = Document(str(file_path))
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Сохраняем заголовки с маркером
            if para.style and para.style.name.startswith("Heading"):
                level = para.style.name.replace("Heading ", "").replace("Heading", "1")
                try:
                    hashes = "#" * int(level)
                except ValueError:
                    hashes = "##"
                parts.append(f"\n{hashes} {text}\n")
            else:
                parts.append(text)

    # Таблицы
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)

    return "\n".join(parts)


# ============================================================
# ODT — odfpy
# ============================================================

def _parse_odt(file_path: Path) -> str:
    """
    Извлечь текст из ODT (OpenDocument Text).

    Использует zipfile + ElementTree напрямую (ODT — это ZIP с content.xml),
    чтобы обойти ошибку ExternalReferenceForbidden в lxml/odfpy.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    # ODT — это ZIP-архив
    try:
        with zipfile.ZipFile(str(file_path), "r") as zf:
            # Основной контент в content.xml
            if "content.xml" not in zf.namelist():
                raise ValueError("content.xml не найден в ODT-архиве")
            xml_bytes = zf.read("content.xml")
    except zipfile.BadZipFile:
        raise ValueError(f"Файл не является валидным ODT/ZIP: {file_path.name}")

    # Парсинг XML без загрузки внешних DTD
    root = ET.fromstring(xml_bytes)

    # ODT namespace для text:p (абзацы) и text:h (заголовки)
    NS = {
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    }

    parts = []

    for elem in root.iter():
        # Убираем namespace из тега для сравнения
        local_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if local_tag in ("p", "h"):
            # Собираем весь текст из элемента и дочерних
            text_content = "".join(elem.itertext()).strip()
            if text_content:
                if local_tag == "h":
                    parts.append(f"\n## {text_content}\n")
                else:
                    parts.append(text_content)

    return "\n".join(parts)


def _odt_get_text(element) -> str:
    """Рекурсивно извлечь текст из ODT-элемента (legacy, для совместимости)"""
    result = ""
    for node in element.childNodes:
        if hasattr(node, "data"):
            result += node.data
        else:
            result += _odt_get_text(node)
    return result


# ============================================================
# Утилиты
# ============================================================

def _decode_bytes(raw_bytes: bytes) -> str:
    """Декодировать байты с автодетекцией кодировки"""
    # Пробуем UTF-8 первым (наиболее частый)
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Пробуем cp1251 (Windows-кириллица)
    try:
        return raw_bytes.decode("cp1251")
    except UnicodeDecodeError:
        pass

    # Фолбэк — chardet
    try:
        import chardet
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8")
        return raw_bytes.decode(encoding, errors="replace")
    except ImportError:
        return raw_bytes.decode("utf-8", errors="replace")


def _clean_text(text: str, is_law: bool = False) -> str:
    """Общая чистка текста после парсинга"""
    # Убираем лишние переносы строк (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Убираем лишние пробелы/табы
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Убираем одиночные номера страниц на отдельных строках.
    # Для законов НЕ применяем: одиночные числа там — номера частей/пунктов.
    if not is_law:
        text = re.sub(r'^\d{1,4}$', '', text, flags=re.MULTILINE)
    # Убираем BOM и нулевые символы
    text = text.replace('\ufeff', '').replace('\x00', '')
    return text.strip()
