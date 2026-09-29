"""
Virality Engine — генерация и выбор заголовков по критериям виральности.

Из 14 критериев (Маркетинг.txt) отобрано 6 для B2B-заголовков:
  1. Фрустрации (боль/блокировка/штраф)
  2. Когнитивные искажения (миф/разоблачение)
  3. Узнаваемая ситуация («у меня тоже так»)
  5. Нереалистичные ожидания (миф о простоте)
  9. Поведенческий хук (вопрос/призыв)
  11. Трансформация (было → стало)

Для каждой темы генерируется 3-5 вариантов заголовка по разным критериям.
Выбор: детерминированный скоринг (цифра + боль + конкретика = баллы).
Если Jev доступен — Jev Choice выбирает финальный вариант.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agents.virality")

# Упрощённые критерии для заголовков (из 14 — только рабочие для B2B)
HEADLINE_CRITERIA = {
    "frustration": {
        "name": "Фрустрация / боль",
        "pattern": "{объект}: {боль} — {решение_кратко}",
        "example": "Блокировка счёта по 115-ФЗ: как разблокировать за 7 дней",
        "score_bonus": 2,  # боль = сильный триггер для B2B
    },
    "myth_busting": {
        "name": "Когнитивное искажение",
        "pattern": "Думаете, {миф}? {правда}",
        "example": "Думаете, 5% НДС всегда выгоднее? Вот когда это ошибочно",
        "score_bonus": 2,
    },
    "relatable": {
        "name": "Узнаваемая ситуация",
        "pattern": "{ситуация} — знакомо? {решение_кратко}",
        "example": "Наняли исполнителя по ГПХ, а он работает как штатный — риск переквалификации",
        "score_bonus": 1,
    },
    "taboo_truth": {
        "name": "Табу-правда",
        "pattern": "{утверждение}, о котором не говорят",
        "example": "Большинство компаний не знают, что уведомление РКН не заменяет политику",
        "score_bonus": 1,
    },
    "transformation": {
        "name": "Трансформация",
        "pattern": "От {проблема} к {результат}: {способ}",
        "example": "От ручного учёта к автоматическому: как сократить отчётность с 3 дней до 2 часов",
        "score_bonus": 1,
    },
    "question_hook": {
        "name": "Поведенческий хук",
        "pattern": "Что делать, если {ситуация}?",
        "example": "Что делать, если банк заблокировал счёт по 115-ФЗ?",
        "score_bonus": 2,
    },
}


def generate_headline_variants(
    topic: str,
    sphere: str = "налоги",
    pain: str = "",
    myth: str = "",
    situation: str = "",
    transformation_from: str = "",
    transformation_to: str = "",
) -> List[Dict[str, Any]]:
    """
    Сгенерировать 3-6 вариантов заголовка по разным критериям виральности.

    Args:
        topic: основная тема статьи
        sphere: домен (налоги / трудовое / защита данных / маркетинг)
        pain: описание боли (для frustration)
        myth: распространённый миф (для myth_busting)
        situation: узнаваемая ситуация (для relatable)
        transformation_from: было (для transformation)
        transformation_to: стало (для transformation)

    Returns:
        [{"criterion": "frustration", "headline": "...", "score": N}, ...]
    """
    variants = []
    topic_short = topic[:80]

    # 1. Фрустрация — всегда доступна (боль есть в любой теме)
    if pain:
        variants.append({
            "criterion": "frustration",
            "headline": f"{pain} — что делать и как избежать",
            "score": HEADLINE_CRITERIA["frustration"]["score_bonus"],
        })
    else:
        variants.append({
            "criterion": "frustration",
            "headline": f"{topic_short}: риски, штрафы и как защитить бизнес",
            "score": HEADLINE_CRITERIA["frustration"]["score_bonus"],
        })

    # 2. Когнитивное искажение — если есть миф
    if myth:
        variants.append({
            "criterion": "myth_busting",
            "headline": f"Думаете, {myth}? Вот почему это ошибочно",
            "score": HEADLINE_CRITERIA["myth_busting"]["score_bonus"],
        })

    # 3. Узнаваемая ситуация — если есть
    if situation:
        variants.append({
            "criterion": "relatable",
            "headline": f"{situation} — знакомо? Разбираем по шагам",
            "score": HEADLINE_CRITERIA["relatable"]["score_bonus"],
        })

    # 4. Табу-правда — генерируем из темы
    variants.append({
        "criterion": "taboo_truth",
        "headline": f"{topic_short}: о чём молчат консультанты",
        "score": HEADLINE_CRITERIA["taboo_truth"]["score_bonus"],
    })

    # 5. Трансформация — если есть данные
    if transformation_from and transformation_to:
        variants.append({
            "criterion": "transformation",
            "headline": f"От {transformation_from} к {transformation_to}: пошаговый план",
            "score": HEADLINE_CRITERIA["transformation"]["score_bonus"],
        })

    # 6. Поведенческий хук — всегда доступен
    variants.append({
        "criterion": "question_hook",
        "headline": f"Что делать, если {topic_short.lower()}?",
        "score": HEADLINE_CRITERIA["question_hook"]["score_bonus"],
    })

    # Сортируем по скору (убывание)
    variants.sort(key=lambda x: -x["score"])
    return variants


def pick_best_headline(
    variants: List[Dict[str, Any]],
    use_jev: bool = True,
) -> Dict[str, Any]:
    """
    Выбрать лучший заголовок: детерминированный скоринг + Jev Choice (если доступен).

    Returns:
        Лучший вариант {"criterion", "headline", "score", "selected_by"}
    """
    if not variants:
        return {}

    # Детерминированный скоринг: добавляем баллы за элементы сильного заголовка
    for v in variants:
        h = v["headline"].lower()
        bonus = 0
        if any(c.isdigit() for c in h):
            bonus += 1  # цифра привлекает внимание
        if "?" in h:
            bonus += 1  # вопрос провоцирует клик
        if len(h) <= 80:
            bonus += 1  # короткий заголовок лучше для SEO
        if any(w in h for w in ["штраф", "риск", "блокировка", "ошибка", "потерять"]):
            bonus += 2  # страх потери — сильнейший триггер B2B
        v["final_score"] = v["score"] + bonus

    # Jev Choice (если доступен)
    if use_jev:
        try:
            from agents.jev_client import get_jev_client, choice_question
            client = get_jev_client()
            if client:
                headlines = {f"opt_{i}": v["headline"] for i, v in enumerate(variants)}
                d = client.decide(
                    state={
                        "task": "Выбери заголовок, наиболее вероятный для клика владельцем МСБ",
                        "variants": headlines,
                    },
                    questions={
                        "best": choice_question(
                            "Какой заголовок вызовет больше всего кликов у владельца малого бизнеса?",
                            {k: None for k in headlines}),
                    },
                )
                best_choice = d.choice("best")
                if best_choice and best_choice in headlines:
                    idx = int(best_choice.split("_")[1])
                    variants[idx]["selected_by"] = "jev"
                    logger.info(f"Virality: Jev выбрал «{variants[idx]['headline'][:60]}»")
                    return variants[idx]
        except Exception as e:
            logger.warning(f"Virality: Jev недоступен ({e}), пробую DeepSeek")

    # DeepSeek Flash (fallback: понимает контекст, дешевле Jev)
    try:
        from openai import OpenAI
        ds_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            timeout=30.0,
        )
        headlines_text = "\n".join(
            f"{i+1}. {v['headline']} (критерий: {v['criterion']})"
            for i, v in enumerate(variants))
        resp = ds_client.chat.completions.create(
            model=os.getenv("MODEL_DEEPSEEK_FLASH", "deepseek-flash"),
            messages=[
                {"role": "system", "content": (
                    "Ты — редактор B2B-издания. Выбери заголовок, который вызовет больше всего "
                    "кликов у владельца малого бизнеса в России. Учитывай: боль, конкретику, "
                    "цифры, страх потери. Верни ТОЛЬКО номер варианта (1, 2, 3...)."
                )},
                {"role": "user", "content": f"Варианты заголовков:\n{headlines_text}\n\nНомер лучшего:"},
            ],
            temperature=0.1, max_tokens=100, timeout=30.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        # Парсим номер
        num_match = re.search(r'\d+', answer)
        if num_match:
            idx = int(num_match.group()) - 1
            if 0 <= idx < len(variants):
                variants[idx]["selected_by"] = "deepseek"
                logger.info(f"Virality: DeepSeek выбрал «{variants[idx]['headline'][:60]}»")
                return variants[idx]
    except Exception as e:
        logger.warning(f"Virality: DeepSeek выбор сбой ({e}), использую скоринг")

    # Детерминированный выбор: максимальный final_score
    best = max(variants, key=lambda x: x.get("final_score", 0))
    best["selected_by"] = "scoring"
    return best
