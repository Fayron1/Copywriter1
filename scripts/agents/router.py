import re
from typing import List, Dict, Any, Tuple
from .patterns import PATTERNS

class ArticleRouter:
    """
    Класс для автоматического выбора типа статьи (роутинга) и 
    динамического наложения SEO-правил на основе входного контракта.
    """

    @staticmethod
    def route(topic: str, description: str, size: str, keywords: List[str], custom_chars: int = 0) -> Dict[str, Any]:
        """
        Принимает входной контракт и возвращает выбранный тип статьи,
        рассчитанные лимиты символов и дополнительные SEO-инструкции.
        """
        desc_lower = (description or "").lower()
        topic_lower = topic.lower()
        
        # 1. Логика выбора типа статьи (роутинг)
        # 1.1 Чек-листы
        if any(w in desc_lower or w in topic_lower for w in ["чек-лист", "check-list", "список", "пунктов", "ошибок", "шагов", "советов"]):
            article_type = "checklist"
        # 1.2 Законы
        elif any(w in desc_lower or w in topic_lower for w in ["закон", "поправки", "изменения в законе", "фз", "нк рф", "ст.", "статья"]):
            article_type = "law_review"
        # 1.3 Кейсы
        elif any(w in desc_lower or w in topic_lower for w in ["кейс", "разбор ситуации", "история компании", "реальный пример", "опыт"]):
            article_type = "case_study"
        # 1.4 Справочники
        elif any(w in desc_lower or w in topic_lower for w in ["справочник", "таблица", "ставки", "штрафы", "сроки", "ставки налогов", "коэффициенты"]):
            article_type = "reference"
        # 1.5 Аналитика (только для лонгридов)
        elif size == "long" and any(w in desc_lower or w in topic_lower for w in ["анализ", "исследование", "рынок", "обзор", "глубокий анализ", "аналитический лонгрид"]):
            article_type = "analysis"
        # 1.6 По умолчанию
        else:
            article_type = "free_style"

        # 2. Получение базовых лимитов из стиля
        pattern = PATTERNS.get(article_type, PATTERNS["free_style"])
        style_target = pattern.get("target_chars", 8000)
        
        # Используем лимиты из st        # 3. Расчет лимитов символов в зависимости от size и custom_chars
        if custom_chars > 0:
            # Если пользователь явно передал объем, используем его без привязки к лимитам стиля
            # Но защищаем от абсурдных значений (от 2000 до 35000)
            target_chars = max(2000, min(35000, custom_chars))
            min_chars = int(target_chars * 0.85)
            max_chars = int(target_chars * 1.15)
        else:
            if size == "short":
                target = 10000
                min_chars = 8500
                max_chars = 11500
            else: # long
                target = 25000
                min_chars = 22000
                max_chars = 28000

        # Адаптивное число пунктов для чек-листа
        num_checklist_items = 10
        if article_type == "checklist":
            if target_chars < 5000:
                num_checklist_items = 5
            elif target_chars < 8000:
                num_checklist_items = 7
            else:
                num_checklist_items = 10

        # 4. Наложение SEO-слоя (только для short)
        seo_instructions = {
            "engineer_instruction": "",
            "heart_instruction": "",
            "sheriff_instruction": ""
        }

        if size == "short" and keywords:
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

        density_config = ArticleRouter.get_density_config(target_chars)

        return {
            "article_type": article_type,
            "target_chars": target_chars,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "seo_instructions": seo_instructions,
            "density_config": density_config,
            "num_checklist_items": num_checklist_items
        }

    @staticmethod
    def route(topic: str, description: str, size: str, keywords: List[str], custom_chars: int = 0) -> Dict[str, Any]:
        """
        Принимает входной контракт и возвращает выбранный тип статьи,
        рассчитанные лимиты символов и дополнительные SEO-инструкции.
        """
        desc_lower = (description or "").lower()
        topic_lower = topic.lower()
        
        # 1. Логика выбора типа статьи (роутинг)
        # 1.1 Чек-листы
        if any(w in desc_lower or w in topic_lower for w in ["чек-лист", "check-list", "список", "пунктов", "ошибок", "шагов", "советов"]):
            article_type = "checklist"
        # 1.2 Законы
        elif any(w in desc_lower or w in topic_lower for w in ["закон", "поправки", "изменения в законе", "фз", "нк рф", "ст.", "статья"]):
            article_type = "law_review"
        # 1.3 Кейсы
        elif any(w in desc_lower or w in topic_lower for w in ["кейс", "разбор ситуации", "история компании", "реальный пример", "опыт"]):
            article_type = "case_study"
        # 1.4 Справочники
        elif any(w in desc_lower or w in topic_lower for w in ["справочник", "таблица", "ставки", "штрафы", "сроки", "ставки налогов", "коэффициенты"]):
            article_type = "reference"
        # 1.5 Аналитика (только для лонгридов)
        elif size == "long" and any(w in desc_lower or w in topic_lower for w in ["анализ", "исследование", "рынок", "обзор", "глубокий анализ", "аналитический лонгрид"]):
            article_type = "analysis"
        # 1.6 По умолчанию
        else:
            article_type = "free_style"

        # 2. Получение базовых лимитов из стиля
        pattern = PATTERNS.get(article_type, PATTERNS["free_style"])
        style_target = pattern.get("target_chars", 8000)
        
        # Используем лимиты из st        # 3. Расчет лимитов символов в зависимости от size и custom_chars
        if custom_chars > 0:
            # Если пользователь явно передал объем, используем его без привязки к лимитам стиля
            # Но защищаем от абсурдных значений (от 2000 до 35000)
            target_chars = max(2000, min(35000, custom_chars))
            min_chars = int(target_chars * 0.85)
            max_chars = int(target_chars * 1.15)
        else:
            if size == "short":
                target = 10000
                min_chars = 8500
                max_chars = 11500
            else: # long
                target = 25000
                min_chars = 22000
                max_chars = 28000

        # Адаптивное число пунктов для чек-листа
        num_checklist_items = 10
        if article_type == "checklist":
            if target_chars < 5000:
                num_checklist_items = 5
            elif target_chars < 8000:
                num_checklist_items = 7
            else:
                num_checklist_items = 10

        # 4. Наложение SEO-слоя (только для short)
        seo_instructions = {
            "engineer_instruction": "",
            "heart_instruction": "",
            "sheriff_instruction": ""
        }

        if size == "short" and keywords:
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

        density_config = ArticleRouter.get_density_config(target_chars)

        return {
            "article_type": article_type,
            "target_chars": target_chars,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "seo_instructions": seo_instructions,
            "density_config": density_config,
            "num_checklist_items": num_checklist_items
        }

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
