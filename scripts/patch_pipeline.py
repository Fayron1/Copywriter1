#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_pipeline.py — патчит pipeline.py на VPS.
Добавляет автоматический fallback gpt-image-2 → dall-e-3.

Запуск: cd /root/Copywriter/scripts && python patch_pipeline.py
"""
import shutil
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "agents" / "pipeline.py"

if not PIPELINE.exists():
    print(f"❌ Файл не найден: {PIPELINE}")
    exit(1)

# Читаем с обработкой кодировки
raw = PIPELINE.read_bytes()
for enc in ["utf-8", "utf-8-sig", "latin-1"]:
    try:
        text = raw.decode(enc)
        print(f"📄 Файл прочитан с кодировкой: {enc} ({len(text)} символов)")
        break
    except UnicodeDecodeError:
        continue
else:
    text = raw.decode("utf-8", errors="replace")
    print(f"⚠️ Файл прочитан с utf-8 + replace ({len(text)} символов)")

# Backup
backup = PIPELINE.with_name("pipeline.py.bak")
shutil.copy2(PIPELINE, backup)
print(f"📦 Бэкап: {backup}")

changes = 0

# ═══════════════════════════════════════════════════════════════
# ПАТЧ 1: Обложка → фолбек
# ═══════════════════════════════════════════════════════════════

# Ищем маркер начала блока обложки
MARKER_COVER_START = 'cover_payload = {'
MARKER_COVER_END = 'cover_data = resp.json()'

if MARKER_COVER_START in text and MARKER_COVER_END in text:
    # Находим строки
    lines = text.split('\n')
    
    # Ищем начало блока (logger.info с "обложку")
    cover_log_idx = None
    cover_payload_idx = None
    cover_data_idx = None
    
    for i, line in enumerate(lines):
        if 'cover_payload = {' in line:
            cover_payload_idx = i
            # Ищем logger.info перед ним
            for j in range(i-1, max(i-5, 0), -1):
                if 'logger.info' in lines[j] and ('обложку' in lines[j] or 'cover' in lines[j].lower()):
                    cover_log_idx = j
                    break
            if cover_log_idx is None:
                cover_log_idx = i  # fallback
        if 'cover_data = resp.json()' in line and cover_payload_idx is not None:
            cover_data_idx = i
            break
    
    if cover_log_idx is not None and cover_data_idx is not None:
        # Определяем отступ
        indent = '            '
        
        fallback_func = [
            f'{indent}def _gen_with_fallback(prompt, size, model="gpt-image-2"):',
            f'{indent}    """Генерация картинки с авто-фолбеком на dall-e-3 при ошибке 400."""',
            f'{indent}    payload = {{"model": model, "prompt": prompt, "size": size, "n": 1}}',
            f'{indent}    try:',
            f'{indent}        logger.info(f"   🚀 Запрос к API изображений (Модель: {{model}}, Размер: {{size}})...")',
            f'{indent}        with httpx.Client(timeout=120.0) as hc:',
            f'{indent}            r = hc.post(f"{{base_url}}/images/generations", json=payload, headers=headers)',
            f'{indent}            r.raise_for_status()',
            f'{indent}            return r.json()',
            f'{indent}    except Exception as e:',
            f'{indent}        if model == "gpt-image-2":',
            f'{indent}            fb_model, fb_size = "dall-e-3", "1792x1024"',
            f'{indent}            logger.warning(f"   ⚠️ Ошибка {{model}} ({{size}}): {{e}}. Фолбек → {{fb_model}} ({{fb_size}})...")',
            f'{indent}            fb_payload = {{"model": fb_model, "prompt": prompt, "size": fb_size, "n": 1}}',
            f'{indent}            with httpx.Client(timeout=120.0) as hc:',
            f'{indent}                r = hc.post(f"{{base_url}}/images/generations", json=fb_payload, headers=headers)',
            f'{indent}                r.raise_for_status()',
            f'{indent}                return r.json()',
            f'{indent}        raise',
            f'',
            f'{indent}logger.info(f"   🚀 Генерирую обложку...")',
            f'{indent}cover_data = _gen_with_fallback(full_cover_prompt, "1536x768")',
        ]
        
        # Заменяем строки от cover_log_idx до cover_data_idx
        lines[cover_log_idx:cover_data_idx+1] = fallback_func
        text = '\n'.join(lines)
        changes += 1
        print("✅ Патч 1/2: Обложка — добавлен фолбек gpt-image-2 → dall-e-3")
    else:
        print(f"⚠️ Патч 1/2: Не удалось найти границы блока обложки")
else:
    print("⚠️ Патч 1/2: Блок обложки не найден")

# ═══════════════════════════════════════════════════════════════
# ПАТЧ 2: Разделители → фолбек
# ═══════════════════════════════════════════════════════════════

MARKER_SEC_START = 'sec_payload = {'
MARKER_SEC_END = 'sec_data = resp.json()'

if MARKER_SEC_START in text and MARKER_SEC_END in text:
    lines = text.split('\n')
    
    sec_log_idx = None
    sec_payload_idx = None
    sec_data_idx = None
    
    for i, line in enumerate(lines):
        if 'sec_payload = {' in line:
            sec_payload_idx = i
            for j in range(i-1, max(i-5, 0), -1):
                if 'logger.info' in lines[j] and ('разделитель' in lines[j] or 'section' in lines[j].lower()):
                    sec_log_idx = j
                    break
            if sec_log_idx is None:
                sec_log_idx = i
        if 'sec_data = resp.json()' in line and sec_payload_idx is not None:
            sec_data_idx = i
            break
    
    if sec_log_idx is not None and sec_data_idx is not None:
        indent = '                '
        
        new_section = [
            f'{indent}logger.info(f"   🚀 Генерирую разделитель {{idx+1}}/{{num_markers}}...")',
            f'{indent}sec_data = _gen_with_fallback(full_section_prompt, "1536x384")',
        ]
        
        lines[sec_log_idx:sec_data_idx+1] = new_section
        text = '\n'.join(lines)
        changes += 1
        print("✅ Патч 2/2: Разделители — добавлен фолбек gpt-image-2 → dall-e-3")
    else:
        print(f"⚠️ Патч 2/2: Не удалось найти границы блока разделителей")
else:
    print("⚠️ Патч 2/2: Блок разделителей не найден")

# ═══════════════════════════════════════════════════════════════
# Сохранение (всегда UTF-8)
# ═══════════════════════════════════════════════════════════════

PIPELINE.write_text(text, encoding="utf-8")

print(f"\n{'═'*55}")
print(f"  Применено патчей: {changes}/2")
print(f"  Файл: {PIPELINE}")
if changes > 0:
    print(f"  Бэкап: {backup}")
    print(f"\n  Запустите генерацию статьи для проверки:")
    print(f"  cd /root/Copywriter/scripts && python generate.py")
print(f"{'═'*55}")
