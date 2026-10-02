"""Sparse-векторы для гибридного поиска (BM25-стиль, stateless).

Проблема (диагностика 2026-10-01): dense e5 размывает точные лексемы —
запрос «ч. 2 ст. 105» должен попадать в чанки ст. 105, но векторная
близость уносит к «жалобам» вообще. Sparse-ветка ловит точные токены,
RRF-слияние объединяет ранги.

Stateless-подход: токен → hash % 2^16 → индекс, вес = tf. Общий словарь
не нужен — тот же хеш и на заливке, и на запросе. Для RRF важны только
ранги, так что IDF не обязателен.

Токены: слова/числа, дефис-составные («44-фз», «223-фз», «п-4»).
"""
from __future__ import annotations

import re
from hashlib import md5
from typing import Dict, List

SPARSE_DIM = 65536  # хеш-пространство; коллизии редки и для RRF не критичны

_TOKEN_RX = re.compile(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*", re.I)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RX.findall((text or "").lower())


def token_index(token: str) -> int:
    return int(md5(token.encode("utf-8")).hexdigest()[:8], 16) % SPARSE_DIM


def sparse_vector(text: str) -> Dict[str, List[float]]:
    """Хеш-sparse текста: {'indices': [...], 'values': [tf, ...]}.

    Формат совместим с qdrant_client.models.SparseVector(**sparse_vector(t)).
    """
    tf: Dict[int, float] = {}
    for t in tokenize(text):
        idx = token_index(t)
        tf[idx] = tf.get(idx, 0.0) + 1.0
    indices = sorted(tf)
    return {"indices": indices,
            "values": [1.0 + (tf[i] - 1.0) * 0.5 for i in indices]}  # мягкий tf: log-подобие без log(0)


def query_sparse(query: str) -> Dict[str, List[float]]:
    """Sparse запроса: tf не важен (однократные вхождения), но унификация
    (те же веса, что у документов) сохраняет монотонность скора."""
    return sparse_vector(query)


# юнит-самопроверка
if __name__ == "__main__":
    sv = sparse_vector("Жалоба по ч. 2 ст. 105 44-ФЗ подаётся в ФАС по 44-фз")
    toks = tokenize("Жалоба по ч. 2 ст. 105 44-ФЗ")
    assert len(toks) == 7 and "44-фз" in toks
    assert len(sv["indices"]) == len(sv["values"]) > 0
    # повтор токена даёт вес >1
    sv2 = sparse_vector("фас фас фас")
    assert sv2["values"][0] == 2.0
    print("sparse_utils: OK,", len(sv["indices"]), "токенов")
