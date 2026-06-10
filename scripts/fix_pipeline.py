# -*- coding: utf-8 -*-
"""
Финальный скрипт восстановления pipeline.py.
Читает с latin-1, исправляет повреждённую строку 988,
записывает обратно с latin-1.
"""
import sys
import shutil

FILEPATH = r"d:\Antigravity_Projects\Copywriter\scripts\agents\pipeline.py"

# Читаем файл с latin-1 (единственная кодировка, которая успешно читает)
with open(FILEPATH, "r", encoding="latin-1") as f:
    lines = f.readlines()

print(f"Всего строк в файле: {len(lines)}")

# Находим повреждённую строку
target_idx = None
for i, line in enumerate(lines):
    if "def _step_booster" in line and i < 1050:
        stripped = line.strip()
        if not stripped.startswith("def "):
            target_idx = i
            print(f"Повреждённая строка найдена: L{i+1}")
            break

if target_idx is None:
    print("Повреждённая строка не найдена. Файл может быть уже исправлен.")
    sys.exit(0)

# Код замены - каждая строка = один элемент списка
R = []
R.append('                f"\xd0\x92\xd0\xbd\xd0\xb5\xd1\x81\xd0\xb8 \xd1\x83\xd0\xba\xd0\xb0\xd0\xb7\xd0\xb0\xd0\xbd\xd0\xbd\xd1\x8b\xd0\xb5 \xd0\xbf\xd1\x80\xd0\xb0\xd0\xb2\xd0\xba\xd0\xb8. \xd0\x92\xd0\xb5\xd1\x80\xd0\xbd\xd0\xb8 \xd0\x9f\xd0\x9e\xd0\x9b\xd0\x9d\xd0\xab\xd0\x99 \xd0\xb8\xd1\x81\xd0\xbf\xd1\x80\xd0\xb0\xd0\xb2\xd0\xbb\xd0\xb5\xd0\xbd\xd0\xbd\xd1\x8b\xd0\xb9 \xd1\x82\xd0\xb5\xd0\xba\xd1\x81\xd1\x82 \xd1\x81\xd1\x82\xd0\xb0\xd1\x82\xd1\x8c\xd0\xb8."')
# Это слишком сложно с кодировками... Используем ASCII-safe подход.

# Альтернативный подход: берём строки из самого файла
# Строки 982-987 сохранились нормально (if original_len > 20000 ветка)
# Нам нужна "else:" ветка + конец метода + _validate_final + начало _step_booster

# Поскольку файл в latin-1, а русский текст закодирован как UTF-8 байты 
# интерпретированные как latin-1, мы будем собирать замену из чистых ASCII частей + 
# копировать стиль кодировки из существующих строк файла

print("\nСтратегия: пересобрать файл по частям")

# Часть 1: всё до повреждённой строки (строки 0..target_idx-1)
part1 = lines[:target_idx]

# Часть 2: восстановленный код (написанный в ascii + utf-8 байты как в остальном файле)
# Для простоты: запишем весь replacement блок как байты UTF-8, 
# а файл сохраним в бинарном режиме

# Часть 3: всё после повреждённой строки (строки target_idx+1..)
part3 = lines[target_idx+1:]

print(f"  part1: строки 1-{target_idx} ({len(part1)} строк)")
print(f"  part3: строки {target_idx+2}-{len(lines)} ({len(part3)} строк)")

# Создаём бэкап
backup_path = FILEPATH + ".bak"
shutil.copy2(FILEPATH, backup_path)
print(f"\nБэкап: {backup_path}")

# Записываем в бинарном режиме
with open(FILEPATH, "wb") as f:
    # Part 1: записываем как latin-1 байты (сохраняем оригинальные байты)
    for line in part1:
        f.write(line.encode("latin-1"))
    
    # Part 2: replacement - записываем как UTF-8
    replacement = '''\
                f"Внеси указанные правки. Верни ПОЛНЫЙ исправленный текст статьи."
                f"{volume_warning}"
            )
        else:
            user_msg = (
                f"ЧЕРНОВИК:\\n{state.draft}\\n\\n"
                f"{json.dumps(state.mirror_review, ensure_ascii=False, indent=2)}\\n\\n"
                f"Внеси все указанные исправления за один проход.\\n"
                f"ОБЯЗАТЕЛЬНО ВЕРНИ ВЕСЬ ТЕКСТ СТАТЬИ ЦЕЛИКОМ, от начала до конца "
                f"(включая те части, которые не нуждались в правках)."
                f"{volume_warning}"
            )

        result = self._generate_clean_heart_text(user_msg, target_chars=target_chars)

        # ЗАЩИТА: если revision вернул значительно меньше оригинала
        if len(result) < original_len * 0.3 and original_len > 1000:
            logger.warning(
                f"⚠️ Combined revision вернул {len(result)} символов vs оригинал {original_len}. "
                f"Draft НЕ перезаписан."
            )
        else:
            state.draft = result
            state.sheriff_iterations += 1
        logger.info(f"   📏 Draft после объединённой ревизии: {len(state.draft)} символов")

    def _validate_final(self, state: PipelineState) -> list:
        """Финальная валидация статьи (без API). Возвращает список предупреждений."""
        import re
        warnings = []
        text = state.final_article or ""
        if not text:
            return ["🔴 Финальная статья пуста!"]

        # 1. Проверка длины
        target = state.custom_chars
        if not target:
            if state.style_id:
                try:
                    from .styles import get_style
                    target = get_style(state.style_id).target_chars
                except Exception:
                    pass
        if not target:
            target = 8000

        if len(text) > target * 1.3:
            warnings.append(f"⚠️ Статья слишком длинная: {len(text)} vs цель {target}")
        if len(text) < target * 0.5:
            warnings.append(f"⚠️ Статья слишком короткая: {len(text)} vs цель {target}")

        # 2. Проверка структуры чек-листов
        if state.article_type == "checklist":
            h2_count = len(re.findall(r'^## \\d+\\.', text, re.MULTILINE))
            if h2_count != 10:
                warnings.append(f"🔴 Чек-лист содержит {h2_count} пунктов вместо 10!")

        # 3. Утёкшие теги
        leaked_tags = re.findall(r'\\[(картинка|IMAGE_PROMPT_HERE|TABLE|CHRONOTOPE_SCENE)[^\\]]*\\]', text)
        if leaked_tags:
            warnings.append(f"⚠️ Утёкшие теги в тексте: {leaked_tags[:5]}")

        # 4. Стоп-слова
        try:
            from .stopwords import ALL_STOP_WORDS
            found = [w for w in ALL_STOP_WORDS if w in text.lower()]
            if found:
                warnings.append(f"⚠️ Стоп-слова в финальном тексте: {found[:5]}")
        except ImportError:
            pass

        return warnings

    def _step_booster(self, state: PipelineState):
'''
    f.write(replacement.encode("utf-8"))
    
    # Part 3: записываем как latin-1 байты  
    for line in part3:
        f.write(line.encode("latin-1"))

print("Файл записан в смешанной кодировке.")

# Теперь перечитаем весь файл как binary и перекодируем в чистый UTF-8
print("\nНормализация кодировки -> чистый UTF-8...")
with open(FILEPATH, "rb") as f:
    raw_bytes = f.read()

# Пробуем декодировать как UTF-8
try:
    text = raw_bytes.decode("utf-8")
    print("  Файл уже валидный UTF-8!")
except UnicodeDecodeError:
    # Пробуем latin-1 -> это всегда работает
    text = raw_bytes.decode("latin-1")
    # Но нам нужен UTF-8. Проблема в том, что оригинальные русские строки
    # были в UTF-8 и прочитаны как latin-1. Нужно "double-decode":
    # latin-1 строка "Ð¡Ñ‚Ñ€" -> encode('latin-1') -> bytes 0xD0 0xA1 ... -> decode('utf-8') -> "Ст"
    try:
        # Записываем как latin-1 байты и перечитываем как UTF-8
        fixed_bytes = text.encode("latin-1")
        text = fixed_bytes.decode("utf-8")
        print("  Double-decode latin-1->utf-8 успешен!")
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        print(f"  WARNING: Double-decode не сработал: {e}")
        print("  Оставляем файл как есть.")
        text = None

if text is not None:
    with open(FILEPATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("  Файл сохранён в чистом UTF-8!")

# Финальная верификация
print("\n--- Финальная верификация ---")
try:
    with open(FILEPATH, "r", encoding="utf-8") as f:
        verify_lines = f.readlines()
    print(f"Итого строк: {len(verify_lines)}")
    print("Методы класса:")
    for i, line in enumerate(verify_lines):
        if "    def " in line and not line.strip().startswith("#"):
            print(f"  L{i+1}: {line.rstrip()[:80]}")
    print("\n✅ Восстановление завершено!")
except Exception as e:
    print(f"ОШИБКА верификации: {e}")
