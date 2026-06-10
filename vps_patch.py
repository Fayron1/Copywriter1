# -*- coding: utf-8 -*-
import re
from pathlib import Path

# Скрипт для безопасного применения патча прямо на VPS сервере
file_path = Path("scripts/agents/pipeline.py")

if not file_path.exists():
    print("ERROR: scripts/agents/pipeline.py not found! Run this script from the root of Copywriter1 directory.")
    exit(1)

# Читаем оригинальный чистый файл в UTF-8
content = file_path.read_text(encoding="utf-8")

# 1. Находим и заменяем метод _step_booster
start_marker = "def _step_booster"
start_idx = content.find(start_marker)
if start_idx != -1:
    while start_idx > 0 and content[start_idx-1] == ' ':
        start_idx -= 1

end_marker = "def _step_artist"
end_idx = content.find(end_marker)
if end_idx != -1:
    while end_idx > 0 and content[end_idx-1] == ' ':
        end_idx -= 1

if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
    new_booster = """    def _step_booster(self, state: PipelineState):
        \"\"\"Шаг 8: Booster — SEO/GEO оптимизация.\"\"\"
        logger.info("🚀 [8/9] Booster: SEO/GEO оптимизация...")
        
        user_msg = (
            f"ТЕМА СТАТЬИ: {state.topic}\\n"
            f"ТИП СТАТЬИ: {state.article_type}\\n"
            f"НАПРАВЛЕНИЕ: {state.direction}\\n\\n"
            f"ЧЕРНОВИК СТАТЬИ:\\n{state.draft}\\n\\n"
            f"- Твой БЮДЖЕТ на SEO-добавки: ровно {state.seo_budget} символов. Это всё, что ты можешь добавить.\\n"
            f"- Citation Bait: вплетай в ПОСЛЕДНЕЕ предложение перед каждым H2 (не создавай новые абзацы).\\n"
            f"- LSI-ключи: перефразируй существующие предложения, не добавляя новых.\\n"
            f"- FAQ: добавь в JSON-поле 'faq' (для Schema.org), но НЕ вставляй блок FAQ в тело статьи.\\n"
            f"- Категорически ЗАПРЕЩЕНО добавлять новые разделы H2/H3.\\n\\n"
            f"Оптимизируй статью и подготовить SEO-пакет."
        )
        
        # Запрашиваем raw текст (без авто-парсинга JSON)
        raw_response = self._call_agent("booster", user_msg, parse_json=False, state=state)
        
        import re
        import json
        
        # 1. Извлекаем JSON-пакет метаданных
        metadata_match = re.search(r'<seo_metadata>\\s*({.*?})\\s*</seo_metadata>', raw_response, re.DOTALL)
        seo_package = {}
        if metadata_match:
            try:
                seo_package = json.loads(metadata_match.group(1).strip())
            except Exception as e:
                logger.warning(f"   ⚠️ Не удалось распарсить JSON в <seo_metadata>: {e}")
                # Попытка спасти данные, убрав markdown-обертки ```json
                cleaned_json = re.sub(r'^```(?:json)?\\s*', '', metadata_match.group(1).strip())
                cleaned_json = re.sub(r'\\s*```$', '', cleaned_json)
                try:
                    seo_package = json.loads(cleaned_json)
                except Exception:
                    pass
        else:
            # Fallback если тегов нет, но весь ответ - это JSON (старый формат)
            try:
                cleaned_json = re.sub(r'^```(?:json)?\\s*', '', raw_response.strip())
                cleaned_json = re.sub(r'\\s*```$', '', cleaned_json)
                seo_package = json.loads(cleaned_json)
            except Exception:
                pass
                
        state.seo_package = seo_package
        
        # 2. Извлекаем оптимизированный текст статьи
        article_match = re.search(r'<optimized_article>\\s*(.*?)\\s*</optimized_article>', raw_response, re.DOTALL)
        optimized_text = ""
        if article_match:
            optimized_text = article_match.group(1).strip()
        else:
            # Fallback если тега <optimized_article> нет, но есть закрывающий </seo_metadata>
            if "</seo_metadata>" in raw_response:
                parts = raw_response.split("</seo_metadata>")
                optimized_text = parts[1].strip()
                optimized_text = re.sub(r'<optimized_article>\\s*', '', optimized_text)
                optimized_text = re.sub(r'</optimized_article>\\s*$', '', optimized_text).strip()
            else:
                # Если в JSON-пакете было поле optimized_text или article_text
                optimized_text = seo_package.get("optimized_text") or seo_package.get("article_text") or ""
                
        # Обновляем статью оптимизированным текстом (или оставляем черновик при неудаче)
        state.final_article = optimized_text if (optimized_text and len(optimized_text) > 100) else state.draft
        
        # Применяем Sanity-постпроцессор очистки артефактов
        state.final_article = self._clean_leaked_ai_artifacts(state.final_article)
        
        # Извлекаем мета-теги
        fallback_meta = _extract_meta_from_text(state.final_article)
        state.final_meta = state.seo_package.get("meta", {})
        for k in ["title", "description", "keywords"]:
            if not state.final_meta.get(k) and fallback_meta.get(k):
                state.final_meta[k] = fallback_meta[k]
                logger.info(f"      📝 Извлечено {k} из текста: '{fallback_meta[k][:50]}...'")
 
        logger.info(f"   📊 Final article: {len(state.final_article)} символов")
        state.steps_completed.append("booster")
"""
    new_booster = new_booster.rstrip() + "\n\n"
    content = content[:start_idx] + new_booster + content[end_idx:]
    print("SUCCESS: _step_booster patched cleanly on VPS!")
else:
    print("ERROR: Could not locate _step_booster block indices.")
    exit(1)

# 2. Обновляем расчет токенов в _call_agent
old_token_str = "calculated_tokens = int(target_chars / 1.8 * 1.45)"
new_token_str = "calculated_tokens = int(target_chars * 0.7)"

if old_token_str in content:
    content = content.replace(old_token_str, new_token_str)
    print("SUCCESS: Token limit successfully updated to target_chars * 0.7")
else:
    token_pattern = r"calculated_tokens = int\(target_chars\s*/\s*1\.8\s*\*\s*1\.45\)"
    content, count_tokens = re.subn(token_pattern, new_token_str, content)
    if count_tokens > 0:
        print("SUCCESS: Token limit successfully updated via regex.")
    else:
        print("WARNING: Token limit calculation string not found in pipeline.py.")

# Записываем обратно в UTF-8
file_path.write_text(content, encoding="utf-8")
print("SUCCESS: pipeline.py patched and saved on VPS!")
