#!/usr/bin/env python3
"""
rebuild_html.py — скрипт для локальной перегенерации article.html из уже готовой папки со статьей.
Использование:
    python scripts/rebuild_html.py 20260521_1224_novye_pravila_i_riski_raboty_с_samozanya
"""
import sys
import json
import logging
from pathlib import Path

# Настройка путей для импорта generate.py и модулей из папки scripts/agents
scripts_dir = Path(__file__).resolve().parent
project_dir = scripts_dir.parent
sys.path.append(str(scripts_dir))
sys.path.append(str(project_dir))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rebuild_html")

try:
    from generate import save_html_preview
except ImportError as e:
    logger.error(f"❌ Не удалось импортировать save_html_preview из generate.py: {e}")
    sys.exit(1)


class MockState:
    """Заглушка объекта State, имитирующая состояние пайплайна для save_html_preview."""
    def __init__(self, final_article, article_type, direction, topic, final_meta, seo_package):
        self.final_article = final_article
        self.draft = None
        self.article_type = article_type
        self.direction = direction
        self.topic = topic
        self.final_meta = final_meta
        self.seo_package = seo_package


def main():
    if len(sys.argv) < 2:
        logger.error("❌ Укажите имя папки со статьей!")
        logger.info("Пример: python scripts/rebuild_html.py 20260521_1224_novye_pravila_i_riski_raboty_s_samozanya")
        sys.exit(1)

    target_input = sys.argv[1]
    # Поддерживаем как относительные/абсолютные пути, так и поиск в стандартных папках вывода
    target_path = Path(target_input)
    
    # 1. Поиск по прямому относительному/абсолютному пути
    # 2. Поиск в корне проекта
    if not target_path.exists():
        target_path = project_dir / target_input
    # 3. Поиск в project/scripts/output (структура на VPS)
    if not target_path.exists():
        target_path = project_dir / "scripts" / "output" / target_input
    # 4. Поиск в project/output
    if not target_path.exists():
        target_path = project_dir / "output" / target_input
    # 5. Поиск в scripts/output относительно папки скрипта
    if not target_path.exists():
        target_path = scripts_dir / "output" / target_input

    if not target_path.exists() or not target_path.is_dir():
        logger.error(f"❌ Директория со статьей не найдена. Проверенные пути:")
        logger.error(f"   - {Path(target_input).resolve()}")
        logger.error(f"   - {(project_dir / target_input).resolve()}")
        logger.error(f"   - {(project_dir / 'scripts' / 'output' / target_input).resolve()}")
        logger.error(f"   - {(scripts_dir / 'output' / target_input).resolve()}")
        sys.exit(1)

    logger.info(f"📂 Анализ папки статьи: {target_path.name}")

    # 1. Проверяем наличие необходимых файлов
    article_md_path = target_path / "article.md"
    seo_json_path = target_path / "seo_package.json"
    debug_json_path = target_path / "pipeline_debug.json"

    if not article_md_path.exists():
        logger.error(f"❌ Файл статьи article.md не найден в {target_path}")
        sys.exit(1)

    # 2. Читаем и парсим article.md (отрезаем YAML frontmatter)
    logger.info("   📄 Чтение article.md...")
    md_content = article_md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if md_content.startswith("---"):
        parts = md_content.split("---", 2)
        if len(parts) >= 3:
            final_article = parts[2].strip()
        else:
            final_article = md_content
    else:
        final_article = md_content

    # 3. Читаем метаданные из seo_package.json
    final_meta = {}
    seo_package = {}
    if seo_json_path.exists():
        logger.info("   🚀 Загрузка seo_package.json...")
        try:
            seo_package = json.loads(seo_json_path.read_text(encoding="utf-8"))
            final_meta = seo_package.get("meta", {})
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка чтения seo_package.json: {e}")
    else:
        logger.warning("   ⚠️ Файл seo_package.json не найден, метаданные будут извлечены из текста.")

    # 4. Читаем базовые параметры из pipeline_debug.json
    topic = "Тема статьи"
    article_type = "analysis"
    direction = "бизнес"
    if debug_json_path.exists():
        logger.info("   🔍 Загрузка pipeline_debug.json...")
        try:
            debug_data = json.loads(debug_json_path.read_text(encoding="utf-8"))
            topic = debug_data.get("topic", topic)
            article_type = debug_data.get("article_type", article_type)
            direction = debug_data.get("direction", direction)
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка чтения pipeline_debug.json: {e}")

    # 5. Собираем MockState
    state = MockState(
        final_article=final_article,
        article_type=article_type,
        direction=direction,
        topic=topic,
        final_meta=final_meta,
        seo_package=seo_package
    )

    # 6. Вызываем оригинальную исправленную функцию генерации HTML
    logger.info("   🌐 Запуск генерации адаптивного HTML...")
    try:
        save_html_preview(state, target_path)
        logger.info(f"   ✅ Успешно! Файл обновлен: {target_path / 'article.html'}")
    except Exception as e:
        logger.error(f"   ❌ Ошибка при генерации HTML: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
