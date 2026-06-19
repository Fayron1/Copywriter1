import re
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from .styles import STYLES

class ArticleRouter:
    """
    Класс для автоматического выбора типа статьи (роутинга) и 
    динамического наложения SEO-правил на основе входного контракта.
    """
    _cache = {}

    @staticmethod
    def route(topic: str, description: str, size: str, keywords: List[str], custom_chars: int = 0) -> Dict[str, Any]:
        """
        Принимает входной контракт и возвращает выбранный тип статьи,
        рассчитанные лимиты символов и дополнительные SEO-инструкции.
        """
        cache_key = (topic, description, size, tuple(keywords or []), custom_chars)
        if cache_key in ArticleRouter._cache:
            return ArticleRouter._cache[cache_key]
        desc_lower = (description or "").lower()
        topic_lower = topic.lower()
        
        # 1. Попытка классификации через легкую LLM (DeepSeek Flash или OpenAI)
        article_type = None
        enrichments = []
        reason = ""
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        deepseek_model = os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-v4-flash")
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("MODEL_OPENAI_TEXT", "gpt-4o")
        
        client = None
        model_name = "deepseek-v4-flash"
        
        if deepseek_key:
            client = OpenAI(api_key=deepseek_key, base_url=deepseek_base)
            model_name = deepseek_model
        elif openai_key:
            client = OpenAI(api_key=openai_key)
            model_name = openai_model

        if client:
            system_prompt = (
                "Ты — интеллектуальный роутер для B2B-контента.\n"
                "Определи основной тип статьи и 1-2 точечных элемента (enrichments) для ее очеловечивания.\n"
                "Доступные типы статей:\n"
                "* checklist: если это чек-лист, список советов или пунктов\n"
                "* law_review: если это разбор законов, поправок, ФЗ, НК РФ\n"
                "* case_study: если это реальный пример, кейс, опыт компании\n"
                "* reference: если это справочник, таблицы ставок, штрафы, лимиты\n"
                "* analysis: если это глубокий анализ, исследование рынка, обзор трендов\n"
                "* free_style: для остальных свободных тем\n\n"
                "Доступные enrichments (точечные вкрапления для очеловечивания):\n"
                "* case_scene: микро-пример из жизни бизнеса\n"
                "* table: структурированная сравнительная таблица\n"
                "* important_box: выделенный блок с предупреждением или советом\n"
                "* faq: раздел часто задаваемых вопросов в конце\n\n"
                "Возвращай ответ строго в формате JSON:\n"
                "{\n"
                "  \"article_type\": \"checklist\",\n"
                "  \"enrichments\": [\"case_scene\"],\n"
                "  \"reason\": \"краткое обоснование на русском языке\"\n"
                "}"
            )
            
            user_msg = (
                f"ТЕМА СТАТЬИ: {topic}\n"
                f"ОПИСАНИЕ: {description or 'нет'}\n"
                f"КЛЮЧЕВЫЕ СЛОВА: {', '.join(keywords) if keywords else 'нет'}\n"
            )
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt + " Ответ строго в формате JSON."},
                        {"role": "user", "content": user_msg}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=15.0
                )
                res_data = json.loads(response.choices[0].message.content)
                article_type = res_data.get("article_type")
                enrichments = res_data.get("enrichments", [])
                reason = res_data.get("reason", "")
            except Exception:
                # В случае ошибки оставляем None для срабатывания fallback-а
                pass

        # 2. Fallback логика выбора типа статьи (keyword-матчинг)
        if not article_type:
            # 2.1 Чек-листы
            if any(w in desc_lower or w in topic_lower for w in ["чек-лист", "check-list", "список", "пунктов", "ошибок", "шагов", "советов"]):
                article_type = "checklist"
            # 2.2 Законы
            elif any(w in desc_lower or w in topic_lower for w in ["закон", "поправки", "изменения в законе", "фз", "нк рф", "ст.", "статья"]):
                article_type = "law_review"
            # 2.3 Кейсы
            elif any(w in desc_lower or w in topic_lower for w in ["кейс", "разбор ситуации", "история компании", "реальный пример", "опыт"]):
                article_type = "case_study"
            # 2.4 Справочники
            elif any(w in desc_lower or w in topic_lower for w in ["справочник", "таблица", "ставки", "штрафы", "сроки", "ставки налогов", "коэффициенты"]):
                article_type = "reference"
            # 2.5 Аналитика (только для лонгридов)
            elif size == "long" and any(w in desc_lower or w in topic_lower for w in ["анализ", "исследование", "рынок", "обзор", "глубокий анализ", "аналитический лонгрид"]):
                article_type = "analysis"
            # 2.6 По умолчанию
            else:
                article_type = "free_style"

        # 3. Получение лимитов из стиля
        style = STYLES.get(article_type, STYLES["free_style"])
        
        # 4. Расчет лимитов символов в зависимости от size и custom_chars
        if custom_chars > 0:
            target_chars = max(2000, min(35000, custom_chars))
            min_chars = int(target_chars * 0.85)
            max_chars = int(target_chars * 1.15)
        else:
            target_chars = style.target_chars
            min_chars = style.min_chars
            max_chars = style.max_chars
            
            # Если размер явно задан как short, ужимаем лимиты больших стилей
            if size == "short" and target_chars > 10000:
                target_chars = 10000
                min_chars = 8500
                max_chars = 11500

        # Адаптивное число пунктов для чек-листа
        num_checklist_items = 10
        if article_type == "checklist":
            if target_chars < 5000:
                num_checklist_items = 5
            elif target_chars < 8000:
                num_checklist_items = 7
            else:
                num_checklist_items = 10

        # 5. Наложение SEO-слоя
        seo_instructions = {
            "engineer_instruction": "",
            "heart_instruction": "",
            "sheriff_instruction": ""
        }

        if keywords:
            primary_key = keywords[0]
            keywords_str = ", ".join([f"'{k}'" for k in keywords])
            
            seo_instructions["engineer_instruction"] = (
                f"\n\n=== ЕДИНЫЙ SEO-СЛОЙ ===\n"
                f"- Обязательно включи в план один H1, содержащий основную ключевую фразу: '{primary_key}'.\n"
                f"- Обязательно включи ключевые слова из списка {keywords_str} как минимум в два заголовка разделов H2 в плане Blueprint.\n"
                f"- Включи блок 'Типичные ошибки бизнеса' или 'Пошаговый алгоритм решения проблемы' (в зависимости от формата).\n"
                f"- В конце статьи обязательно спроектируй блок 'FAQ' (Часто задаваемые вопросы) из 3-5 вопросов и коротких ответов, соответствующих реальным поисковым запросам по теме."
            )
            
            seo_instructions["heart_instruction"] = (
                f"\n\n=== ЕДИНЫЙ SEO-СЛОЙ ===\n"
                f"- Используй ключевую фразу '{primary_key}' строго в заголовке H1.\n"
                f"- Распредели ключевые фразы {keywords_str} по тексту. Они должны встретиться:\n"
                f"  1) В первых 1000 символах статьи (во вступлении).\n"
                f"  2) Как минимум в двух подзаголовках H2.\n"
                f"  3) В формулировках вопросов в блоке FAQ в конце статьи.\n"
                f"- Ввод ключевых слов должен быть естественным, без спама и грамматических несогласований."
            )
            
            seo_instructions["sheriff_instruction"] = (
                f"\n\n=== ЕДИНЫЙ SEO-СЛОЙ (Проверка) ===\n"
                f"1. Проверь наличие H1 с ключевой фразой '{primary_key}'.\n"
                f"2. Проверь вхождение ключевых слов {keywords_str} в введении (первые 1000 символов) и в двух H2.\n"
                f"3. Убедись, что блок FAQ присутствует в конце статьи и содержит 3-5 вопросов с ответами.\n"
                f"4. Проверь естественность вхождения ключевых слов."
            )

        # 6. Внедрение enrichments для очеловечивания
        if enrichments:
            enrichment_instructions_eng = "\n\n=== ДОПОЛНИТЕЛЬНЫЕ ЭЛЕМЕНТЫ ОЧЕЛОВЕЧИВАНИЯ (ENRICHMENTS) ===\n"
            enrichment_instructions_heart = "\n\n=== ДОПОЛНИТЕЛЬНЫЕ ЭЛЕМЕНТЫ ОЧЕЛОВЕЧИВАНИЯ (ENRICHMENTS) ===\n"
            
            for el in enrichments:
                if el == "case_scene":
                    enrichment_instructions_eng += "- Обязательно запланируй один раздел с живым микро-кейсом (бизнес-иллюстрация из жизни компании с именами сотрудников).\n"
                    enrichment_instructions_heart += "- Напиши живой микро-кейс (описание конкретной практической ситуации из жизни бизнеса с именами участников и коротким диалогом).\n"
                elif el == "table":
                    enrichment_instructions_eng += "- Запланируй структурированную сравнительную таблицу в одном из содержательных разделов.\n"
                    enrichment_instructions_heart += "- Включи в текст сравнительную Markdown-таблицу для наглядности.\n"
                elif el == "important_box":
                    enrichment_instructions_eng += "- Запланируй важный совет или предупреждение в виде отдельного логического блока.\n"
                    enrichment_instructions_heart += "- Оформи ключевой совет или важное предостережение в виде Markdown-блока цитирования '> Важно: ...'.\n"
                elif el == "faq":
                    enrichment_instructions_eng += "- Спроектируй в конце статьи блок FAQ (Часто задаваемые вопросы) из 3-5 вопросов.\n"
                    enrichment_instructions_heart += "- Напиши в конце статьи раздел FAQ (Часто задаваемые вопросы) с 3-5 актуальными вопросами и емкими ответами.\n"
            
            seo_instructions["engineer_instruction"] += enrichment_instructions_eng
            seo_instructions["heart_instruction"] += enrichment_instructions_heart

        density_config = ArticleRouter.get_density_config(target_chars)

        result = {
            "article_type": article_type,
            "target_chars": target_chars,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "seo_instructions": seo_instructions,
            "density_config": density_config,
            "num_checklist_items": num_checklist_items,
            "enrichments": enrichments,
            "routing_reason": reason
        }
        ArticleRouter._cache[cache_key] = result
        return result

    @staticmethod
    def get_density_config(target_chars: int) -> Dict[str, str]:
        """
        Возвращает параметры плотности текста и стиль прозы в зависимости от объема.
        """
        if target_chars < 5000:
            return {
                "sentences": "2-3",
                "hook_size": "1 короткий абзац",
                "tone_style": "предельно лаконичная, емкая бизнес-проза без воды. 1 абзац = 1 четкий факт/мысль",
                "h2_count": "2-3"
            }
        elif target_chars < 8000:
            return {
                "sentences": "3-4",
                "hook_size": "1-2 абзаца",
                "tone_style": "сбалансированный инфостиль, плавное чередование фактов",
                "h2_count": "3-4"
            }
        elif target_chars <= 10000:
            return {
                "sentences": "3-5",
                "hook_size": "2-3 абзаца",
                "tone_style": "развернутая журнальная проза, подробные практические детали",
                "h2_count": "4-5"
            }
        else:
            return {
                "sentences": "4-6",
                "hook_size": "полноценный подраздел",
                "tone_style": "глубокий аналитический стиль с причинно-следственными связями",
                "h2_count": "5-7"
            }
