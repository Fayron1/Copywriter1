"""
Клиент Jev AI (https://jev-ai.pro) — вероятностный decision-слой.

TypeSafe-совместимый hosted-эндпойнт:
  POST {base}/v1/systemone  — основной decision API
  GET  {base}/v1/models     — список моделей (без инференса)

Контракт (из https://jev-ai.pro/docs):
- Аутентификация: Authorization: Bearer <JEV_AI_API_KEY>.
- base_url = https://jev-ai.pro/api (БЕЗ /v1 и /systemone — клиент добавляет сам;
  «поменять только ключ» недостаточно — сервис выбирает baseURL).
- Вопросы: noul (да/нет, ответ 0..1), choice (2-255 опций), score (2-10 уровней).
- Ретраи: только 429 (с Respect-After) и 502/503 (bounded backoff).
  422 — не ретраить (чинить запрос). 504 — исход неопределён: не более одной
  повторной попытки. 401/402 — конфигурация/баланс, ретраи бессмысленны.
- Лимиты: 1000 запросов/мин, тело <= 256000 байт, <= 64 вопросов, id <= 64 симв.

Ключ JEV_AI_API_KEY живёт ТОЛЬКО в серверном окружении (.env на VPS).
Модуль не импортируется браузерным кодом; в логи ключ не пишется.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger("agents.jev")

DEFAULT_BASE_URL = "https://jev-ai.pro/api"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 30.0

MAX_BODY_BYTES = 256_000
MAX_QUESTIONS = 64
MAX_QUESTION_ID = 64

# Ретраи: только для транзиентных кодов; количество попыток всего — 3.
_RETRYABLE_STATUS = {429, 502, 503}
_RETRY_DELAYS = [1.0, 3.0]
# 504: исход таймаута неопределён (ответ мог быть засчитан) — максимум один ретрай.
_TIMEOUT_RETRY_DELAY = 2.0


# ============================================================
# Ошибки (по таблице #errors из доков)
# ============================================================

class JevError(Exception):
    """Базовая ошибка Jev AI."""

    status: Optional[int] = None

    def __init__(self, message: str, status: Optional[int] = None,
                 body: Optional[Dict] = None):
        super().__init__(message)
        self.status = status if status is not None else self.status
        self.body = body or {}


class JevAuthError(JevError):
    """401: ключ отсутствует/невалиден/отозван — проверить Bearer и ключ."""
    status = 401


class JevBalanceError(JevError):
    """402: баланс исчерпан — проверить биллинг."""
    status = 402


class JevValidationError(JevError):
    """422: невалидные body/state/model/questions — чинить запрос, НЕ ретраить."""
    status = 422


class JevTimeoutError(JevError):
    """504: таймаут upstream; исход неопределён до проверки outcome."""


class JevUnreachableError(JevError):
    """Сеть/DNS/соединение — транзиентно, допустим bounded retry."""


# ============================================================
# Конструкторы вопросов
# ============================================================

def noul_question(instructions: str) -> Dict[str, Any]:
    """Да/нет вопрос. Ответ: вероятность «да» 0..1 в поле answers.<id>.noul."""
    return {"type": "noul", "instructions": instructions}


def choice_question(instructions: str, options: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Выбор одного из 2-255 вариантов.

    options: {option_id: описание или None}. Ответ содержит choice
    (выбранный id), probabilities по всем опциям и confidence.
    """
    if not (2 <= len(options) <= 255):
        raise ValueError(f"choice: нужно 2-255 опций, получено {len(options)}")
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {k: (v if v is not None else None) for k, v in options.items()},
    }


def score_question(instructions: str, levels: List[str]) -> Dict[str, Any]:
    """Оценка по 2-10 упорядоченным уровням. Ответ: числовой score + распределение."""
    if not (2 <= len(levels) <= 10):
        raise ValueError(f"score: нужно 2-10 уровней, получено {len(levels)}")
    return {"type": "score", "instructions": instructions, "levels": list(levels)}


# ============================================================
# Ответ
# ============================================================

class JevAnswer:
    """Типизированный доступ к одному ответу."""

    __slots__ = ("question_id", "type", "raw")

    def __init__(self, question_id: str, raw: Dict[str, Any]):
        self.question_id = question_id
        self.type = raw.get("type")
        self.raw = raw

    @property
    def value(self) -> Union[str, float, None]:
        """choice -> выбранный id; noul/score -> float."""
        if self.type == "choice":
            return self.raw.get("choice")
        if self.type in ("noul", "score"):
            return self.raw.get(self.type)
        return None

    @property
    def yes_probability(self) -> Optional[float]:
        """Для noul: вероятность «да»."""
        return self.raw.get("noul") if self.type == "noul" else None

    @property
    def probabilities(self) -> Dict[str, float]:
        return self.raw.get("probabilities") or {}

    @property
    def confidence(self) -> Optional[float]:
        return self.raw.get("confidence")

    def __repr__(self) -> str:  # pragma: no cover
        return f"JevAnswer({self.question_id}={self.value!r}, conf={self.confidence})"


class JevDecision:
    """Результат одного вызова systemOne."""

    __slots__ = ("model", "answers", "usage", "raw")

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.model = raw.get("model", "")
        self.usage = raw.get("usage") or {}
        self.answers: Dict[str, JevAnswer] = {
            qid: JevAnswer(qid, a) for qid, a in (raw.get("answers") or {}).items()
        }

    def get(self, question_id: str) -> Optional[JevAnswer]:
        return self.answers.get(question_id)

    def choice(self, question_id: str, default: Optional[str] = None) -> Optional[str]:
        ans = self.answers.get(question_id)
        if ans and ans.type == "choice":
            return ans.raw.get("choice", default)
        return default

    def yes(self, question_id: str, default: Optional[float] = None) -> Optional[float]:
        ans = self.answers.get(question_id)
        if ans and ans.type == "noul":
            return ans.raw.get("noul", default)
        return default

    def score(self, question_id: str, default: Optional[float] = None) -> Optional[float]:
        ans = self.answers.get(question_id)
        if ans and ans.type == "score":
            return ans.raw.get("score", default)
        return default

    def __repr__(self) -> str:  # pragma: no cover
        return f"JevDecision(model={self.model!r}, answers={list(self.answers)})"


# ============================================================
# Клиент
# ============================================================

class JevClient:
    """
    REST-клиент Jev AI (эквивалент TypeSafeClient из @typesafe-ai/sdk).

    >>> client = JevClient()                      # ключ/base/model из env
    >>> client.list_models()                      # GET /v1/models, без инференса
    >>> d = client.decide(
    ...     state={"topic": "Изменения НК РФ"},
    ...     questions={
    ...         "article_type": choice_question(
    ...             "Тип статьи?",
    ...             {"law_review": None, "analysis": None, "other": None}),
    ...         "needs_law_check": noul_question("Нужен ли факт-чек законов?"),
    ...     })
    >>> d.choice("article_type"), d.yes("needs_law_check")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = (api_key or os.getenv("JEV_AI_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("JEV_API_BASE", DEFAULT_BASE_URL)).strip().rstrip("/")
        self.model = (model or os.getenv("JEV_MODEL", DEFAULT_MODEL)).strip()
        self.timeout = timeout
        self._session: Optional[requests.Session] = None

    # -- жизненный цикл -------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise JevAuthError("JEV_AI_API_KEY не задан (серверное окружение/.env)")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    # -- валидация тела по докам ---------------------------------------

    @staticmethod
    def _validate(state: Any, questions: Dict[str, Dict[str, Any]]) -> None:
        if not questions:
            raise JevValidationError("questions обязателен (1-64 вопросов)")
        if len(questions) > MAX_QUESTIONS:
            raise JevValidationError(f"Максимум {MAX_QUESTIONS} вопросов, передано {len(questions)}")
        for qid in questions:
            if not qid or len(qid) > MAX_QUESTION_ID:
                raise JevValidationError(f"id вопроса должен быть 1-{MAX_QUESTION_ID} символов: {qid!r}")
        if state in (None, "", [], {}):
            raise JevValidationError("state обязателен и непуст")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)
        return self._get_session().request(
            method, url, json=payload, headers=headers, timeout=self.timeout,
        )

    # -- API ------------------------------------------------------------

    def list_models(self) -> List[Dict[str, str]]:
        """GET /v1/models — без инференса, баланса не тратит."""
        resp = self._request("GET", "/v1/models")
        if resp.status_code != 200:
            raise self._map_error(resp)
        return resp.json().get("models", [])

    def decide(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> JevDecision:
        """
        POST /v1/systemone — основной decision-вызов.

        state: строка/объект/массив (непустой).
        questions: {question_id: noul_question()/choice_question()/score_question()}.
        Ретраи только для 429 (Retry-After) и 502/503 (backoff); 504 — один ретрай.
        """
        self._validate(state, questions)
        body = {
            "model": model or self.model,
            "state": state,
            "questions": questions,
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_BODY_BYTES:
            raise JevValidationError(
                f"Тело запроса {len(encoded)} байт превышает лимит {MAX_BODY_BYTES}"
            )

        attempt = 0
        timeout_retried = False
        while True:
            try:
                resp = self._request("POST", "/v1/systemone", payload=body)
            except requests.RequestException as e:
                # Сетевые сбои транзиентны — bounded retry, как для 502/503.
                if attempt < len(_RETRY_DELAYS):
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(f"Jev: сетевая ошибка ({e}), ретрай через {delay}с")
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise JevUnreachableError(f"Jev недоступен: {e}") from e

            if resp.status_code == 200:
                return JevDecision(resp.json())

            if resp.status_code in _RETRYABLE_STATUS:
                if attempt < len(_RETRY_DELAYS):
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else _RETRY_DELAYS[attempt]
                    except ValueError:
                        delay = _RETRY_DELAYS[attempt]
                    delay = max(0.5, min(delay, 30.0))
                    logger.warning(
                        f"Jev: HTTP {resp.status_code}, ретрай {attempt + 1} через {delay}с"
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise self._map_error(resp)

            if resp.status_code == 504 and not timeout_retried:
                # Исход неопределён: ответ мог быть засчитан. Один ретрай, с предупреждением.
                timeout_retried = True
                logger.warning("Jev: HTTP 504 (таймаут upstream), одна повторная попытка")
                time.sleep(_TIMEOUT_RETRY_DELAY)
                continue

            raise self._map_error(resp)

    # -- маппинг ошибок --------------------------------------------------

    @staticmethod
    def _map_error(resp: requests.Response) -> JevError:
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text[:500]}
        detail = body.get("detail") or body.get("error") or body.get("message") or body

        status = resp.status_code
        text = f"Jev HTTP {status}: {detail}"
        if status == 401:
            return JevAuthError(text, status, body)
        if status == 402:
            return JevBalanceError(text, status, body)
        if status == 422:
            return JevValidationError(text, status, body)
        if status == 504:
            return JevTimeoutError(text, status, body)
        return JevError(text, status, body)


# ============================================================
# Фабрика для пайплайна: один клиент на процесс + graceful degradation
# ============================================================

_client: Optional[JevClient] = None


def get_jev_client() -> Optional[JevClient]:
    """Клиент из env; None + warning, если ключ не настроен (RAG-стиль: не ронять пайплайн)."""
    global _client
    if _client is None:
        _client = JevClient()
        if not _client.configured:
            logger.info("Jev: JEV_AI_API_KEY не задан — decision-слой отключён")
            return None
    return _client


def jev_enabled(feature_flag: str = "JEV_ROUTER_ENABLED") -> bool:
    """Фича-флаг + наличие ключа. По умолчанию включено, отключается '=0'."""
    flag = os.getenv(feature_flag, "1").strip().lower()
    return flag not in ("0", "false", "no", "off") and get_jev_client() is not None


# ============================================================
# CLI: проверка и демо-вызов
#   python scripts/agents/jev_client.py --models
#   python scripts/agents/jev_client.py --demo
# ============================================================

def _load_local_env() -> None:
    """Автономный запуск: подтянуть .env из корня проекта (в пайплайне его грузит generate.py).

    Без зависимости от python-dotenv: если пакет недоступен (например, системный
    python3 на VPS), парсим .env вручную.
    """
    from pathlib import Path

    candidates = (
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    )
    env_file = next((p for p in candidates if p.exists()), None)
    if env_file is None:
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Jev AI client — проверка и демо")
    parser.add_argument("--models", action="store_true", help="GET /v1/models (без инференса)")
    parser.add_argument("--demo", action="store_true",
                        help="Один маленький decision-вызов (тратит баланс)")
    args = parser.parse_args()

    _load_local_env()
    client = JevClient()
    if not client.configured:
        raise SystemExit("JEV_AI_API_KEY не задан — заполни .env")

    if args.models or not args.demo:
        print("Models:", json.dumps(client.list_models(), ensure_ascii=False, indent=2))

    if args.demo:
        decision = client.decide(
            state={
                "topic": "Какие изменения ждут НК РФ с 2027 года",
                "description": "Разбор поправок в налоговое законодательство для B2B-аудитории",
                "keywords": ["НК РФ", "налоги 2027"],
            },
            questions={
                "article_type": choice_question(
                    "Определи тип статьи по теме и описанию.",
                    {
                        "law_review": "Разбор законов, поправок, ФЗ, НК РФ",
                        "analysis": "Глубокий анализ, обзор трендов",
                        "reference": "Справочник, таблицы ставок, лимиты",
                        "checklist": "Чек-лист, список советов",
                        "case_study": "Кейс, опыт компании",
                        "free_style": "Свободная тема",
                    },
                ),
                "needs_law_freshness_check": noul_question(
                    "Требует ли тема обязательной сверки с актуальной редакцией законодательства?"
                ),
            },
        )
        print("Decision:", json.dumps(decision.raw, ensure_ascii=False, indent=2))
        print("Модель назначения:", decision.model)
        print("article_type =", decision.choice("article_type"),
              "| confidence =", decision.get("article_type").confidence if decision.get("article_type") else None)
        print("needs_law_freshness_check =", decision.yes("needs_law_freshness_check"))
        print("usage =", decision.usage)
