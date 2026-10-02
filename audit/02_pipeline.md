# Аудит Copywriter1 — Этап 2: `pipeline.py` + стадии Claim Ledger / Registry

**Дата:** 2026-10-02
**Объём:** `scripts/agents/pipeline.py` — прочитан целиком (5 489 строк), `claim_ledger.py` (275), `registry.py` (377), `prompts.py` (+проверка part1/2/3 на стыках), `virality_engine.py` (структура и интеграция).
**Метод:** полное чтение кода + сверка контрактов вызовов по сигнатурам + проверка свежего незакоммиченного диффа (+266 строк) + контрольные regex-проверки кодом.
**Влияние на систему:** ноль, только чтение.

---

## 1. Как устроен конвейер (фактическая карта, не из документации)

`Pipeline.run()` (строка 561) → этапы в порядке вызова:

```
Brain(936) → Fact-Finder(955) → Freshness(1178) → Fact-Verify(1197) → Scout(1239, опц.)
→ Engineer(1301) → Plan-Critique(1402) → Heart(1529, до 3 попыток)
→ Claim-Check(1771) → Validate-References(1852) → Temporal-Check(1924)
→ Quality-Edit-Loop(2475, 3 итерации + эскалация)
→ Booster(4168) → Statistical-Humanize(3965) → Smart-Hard-Cut(4584)
→ волна детерминированных проверок (725–859) → Assemble-References(1988)
→ Deliverables(2104: CTA + контент-пакет) → Artist(4287, опц.)
```

Состояние — `PipelineState` (dataclass, 269). Ключевые потоки: `facts` → `claims/claim_pack` (claim_ledger) → `rag_block` Heart; `norm_params` (строгие темы) → `format_params_block` → Heart/Booster/Reviewer; `draft` → quality loop → `final_article`.

**Система «работает» подтверждается структурой:** все этапы деградируют через try/except с логом, ни один сбой вспомогательного этапа не роняет пайплайн, кроме штатного fail-fast (0 фактов дважды → RuntimeError, строка 1083 — это намеренный гвард).

---

## 2. 🔴 Находки среднего приоритета (влияют на результат/гейт)

### 2.1. Claim pack никогда не подтверждает числа в H1 — `pipeline.py:816–819`

```python
cp_nums = set(_re_cp.sub(r"\s", "", state.claim_pack).split())
```

`sub(r"\s", "")` удаляет **весь** whitespace из claim pack → `.split()` возвращает **один гигантский токен** (весь pack без пробелов). В `contradiction_checker.check_h1_support(article, claim_pack_numbers)` аргумент ожидается как **множество строк-чисел** (`allowed = body_nums | (claim_pack_numbers or set())`, далее `n not in allowed`).

- Последствие: ветка «число есть в claim pack» никогда не срабатывает → ложные `🔴 ЧИСЛО В H1 БЕЗ ПОДТВЕРЖДЕНИЯ` для чисел, которые в pack есть, но в теле перефразированы. Это прямые ложные красные флаги в publish-gate.
- Направление фикса: извлекать числа регэкспом (`re.findall(r"\d[\d\u00a0 ]*(?:[.,]\d+)?", claim_pack)` с той же нормализацией, что `_numbers` в contradiction_checker), а не split.

### 2.2. Booster может затереть H2-заголовок вместо H1 — `pipeline.py:4274–4282`

```python
h_match = re.search(r"^(#{1,2}\s+.*?)$", state.final_article, re.MULTILINE)
...
state.final_article = state.final_article.replace(h_line, f"# {best_title}", 1)
```

Паттерн матчит **и `#`, и `##`**. Если в статье нет H1 (Heart начал с H2 — редкий, но возможный выход модели), первой найденной строкой будет H2 первого раздела, и она **заменяется** на `# {best_title}`: текст настоящего заголовка раздела теряется, структура (H2-счётчики, TOC, references) сдвигается.
- Направление фикса: сначала искать `^#\s`, и только при отсутствии — prepend, как уже сделано в ветке else.

### 2.3. IndexError от галлюцинации ревизора — `pipeline.py:2626–2627`

```python
for idx in instr_by_idx:      # section_index из ответа LLM
    sec = sections[idx]        # БЕЗ проверки границ
```

В основной петле Surgical Edit Loop `section_index` от внешнего ревизора (Gemini, помечен в registry как «нестабильная») конвертируется в int и используется для индексации **без** проверки `0 <= idx < len(sections)`. Проверка есть только в эскалации (`pipeline.py:2711`). Галлюцинация индекса = необработанный IndexError → весь пайплайн падает в `failed` на поздней стадии (после Heart — потеря всей работы).
- Направление фикса: фильтр `0 <= idx < len(sections) and sections[idx]["level"] == 2` рядом с `int(idx)` (как это уже сделано в `_build_edit_plan:3669`).

### 2.4. Red Team видит только первые 12 000 символов — `pipeline.py:2227`

```python
f"ТЕКСТ СТАТЬИ:\n{draft[:12000]}\n\n"
```

Для лонгридов (20–30k символов) 60% статьи **вне adversarial-аудита**. Red Team — заявленный барьер «докажи, что публиковать нельзя», а проверяет фактически начало. Для сравнения: обычный ревизор в `_evaluate_draft_score` получает `draft` целиком (строка 2279).
- Направление фикса: либо передавать весь текст (ревизор и так получает), либо разбивать на 2 вызова. [СПОРНО] — возможно, осознанный бюджетный компромисс, но в коде/доках нигде не отмечено.

### 2.5. Битая ссылка на обложку при сбое Artist — `pipeline.py:4563–4570`

Встраивание `![Обложка](images/main.png)` стоит **вне** try-блока генерации (4410–4559). Если KIE-ключ не задан / API упал / `_save_image_from_response` вернул False (все пути дают `raise ValueError` → catch на 4559 → «продолжаем без картинок»), H1-embed всё равно выполняется → в финальной статье остаётся ссылка на несуществующий `images/main.png`. Маркеры разделов при этом вычищаются корректно (4573–4579), а обложка — нет.
- Стреляет при `skip_images=False` (не дефолт) + сбое API.
- Направление фикса: флаг успешного сохранения обложки, embed только при True.

### 2.6. Налоговый калькулятор всегда считается для «доходов» — `pipeline.py:1580–1584`

```python
revenue = int(rev_m.group(1)) * 1_000_000 if rev_m else 120_000_000
calc_results = calc_comparison(
    revenue=revenue, purchases=revenue * 0.65,
    includes_vat=False, legal_form="IP", usn_object="income")
```

Единственный параметр, извлекаемый из темы — выручка. `purchases` (65% выручки), `includes_vat=False`, `legal_form="IP"`, `usn_object="income"` — жёсткий дефолт. Для темы «УСН доходы минус расходы» / «НДС у ООО» Heart получает таблицу сравнения, посчитанную **для другого объекта налогообложения и другого юрлица**, и по доктрине «ОБЯЗАН использовать её дословно». Доля закупок 65% может вообще не соответствовать кейсу из ТЗ.
- [СПОРНО] в части намеренности: комментарий в коде называет это «дефолтный кейс (120 млн, закупки 78 млн)», т.е. дизайн-решение. Но для режима УСН выбор `usn_object="income"` при теме про «доходы минус расходы» — фактическая ошибка подбора сценария.
- Направление фикса: определять `usn_object` по теме/ТЗ (как минимум матчить «доходы минус расходы»), долю закупок брать из description.

### 2.7. Рассинхрон года в паспорт-блоке — `pipeline.py:1641`, `2259`, `4179`

```python
rag_block = format_params_block(_params, datetime.datetime.now().year) + ...
```

Паспорт строится на `state.article_year` (строка 1121, тема «в 2027» → паспорт 2027), а блок для Heart/Red-Reviewer/Booster подписывается текущим годом (`datetime.now().year`). Для статьи про 2027 модель видит заголовок «ЗАФИКСИРОВАННЫЕ ПАРАМЕТРЫ НА 2026 ГОД» со значениями 2027 — прямой приглашение к смешению годов, которое система всю дорогу ловит. `enforce_params` при этом вызывается с правильным `article_year` (строка 750) — несогласованность внутри одного механизма.
- Направление фикса: передавать `getattr(state, "article_year", now.year)` во всех трёх местах.

---

## 3. 🟡 Находки низкого приоритета

| # | Место | Суть |
|---|---|---|
| 3.1 | `pipeline.py:2063` | `.lstrip("www.")` в дедупе доменов: lstrip срезает **набор символов** {'w','.'} слева. `web.example.ru` → `eb.example.ru`, `www.wtf.ru` → `tf.ru`. Дедуп доменов работает неточно. Фикс: `if host.startswith("www."): host = host[4:]` |
| 3.2 | `pipeline.py:3013–3014, 5086–5087` | `_heart_chain_idx` персистентен на инстансе Pipeline: модель, упавшая однажды, не возвращается в ротацию до конца жизни объекта. Для одного прогона — фича; при переиспользовании Pipeline на серию статей — деградация |
| 3.3 | `pipeline.py:725–859` | Один общий try на всю «волну проверок»: исключение в `enforce_params` молча глотает ВСЕ последующие валидаторы (matrix, form-deadline, year-rules, truncation, math, contradictions, events, entities) с одним warning. Единая точка отказа пачки |
| 3.4 | `claim_ledger.py:248–250` | `audit_coverage`: годы (2026) и числа-строки ≥2 симв. без исключений попадают в «НОВЫЕ ЧИСЛА» → систематический шум 🟡 в publish-gate. В `contradiction_checker` годы исключены (`_YEAR`), здесь нет |
| 3.5 | `claim_ledger.py:51` | `%\b` в паттерне `numeric`: `re.search(...'20% компаний')` → **False** (проверено кодом: `%` не-словесный символ, `\b` после него на границе с пробелом не возникает). Проценты не маркируются как verifiable-паттерн |
| 3.6 | `registry.py:57` | Поле `retry_on_fail` **никем не читается** (grep по scripts: только registry). Комментарий «retry_on_fail поднят до 4» не имеет эффекта: реальные ретраи — `AGENT_MAX_RETRIES` (дефолт 3) в `_chat_completion` + attempt-loop в `_call_agent` |
| 3.7 | `virality_engine.py` | **Не импортируется никем** (grep по всем scripts). Headline engine из коммитов d7aa2cd/9df8e26 не подключён ни к pipeline, ни к topic_ranker/topic_generator — мёртвый код или незавершённая интеграция |
| 3.8 | `pipeline.py:2304–2473` | `_step_heart_best_of` не вызывается (мёртвый код ~170 строк). Внутри неверная подпись: комментарий и лог «аварийный fallback на DeepSeek Pro», фактически `models_pool[0]` = Claude Opus через KIE (строка 2371) |
| 3.9 | `pipeline.py` (docstring 18–20) | Legacy-методы `_step_sheriff`, `_step_mirror`, `_heart_patch`, `_step_heart_revision`, `_step_heart_humanize`, `_step_combined_revision`, `_build_edit_plan`, `_raw_json_call`, `_get_sheriff_guidance`, `_heart_expand` — не вызываются из активного пути (задокументировано). ~700 строк. Жив только `_heart_condense` (вызов из `_apply_smart_hard_cut:4618`) |
| 3.10 | `pipeline.py:612` | `preflight()` всегда с `provider="deepseek"`: KIE-модели (включая внешнего ревизора `gemini-3-8-flash-openai`) на старте не проверяются никогда, если явно не `--provider kie`. Стабильность ревизора остаётся на совести рантайма |
| 3.11 | `pipeline.py:662` | Мojibake в логе: `ð` вместо эмодзи (битая UTF-8 в исходнике) |
| 3.12 | `pipeline.py:5072+5150` | `max_tokens_for_call` вычисляется дважды, первая версия — мёртвая |
| 3.13 | `PipelineState` | Поля `claims`, `claim_pack`, `norm_params`, `article_year`, `final_warnings`, `facts_filtered_audit`, `engineer_user_msg`, `last_call_truncated` не объявлены в dataclass — контракт состояния невидим (работает, но рефакторинг опасен) |
| 3.14 | `pipeline.py:920–924` | `budget_exhausted` детектится только по подстрокам "insufficient"/"402" — формулировки баланса DeepSeek могут отличаться, статус будет `failed` без специфики |

---

## 4. Что проверено и чисто

- **`run()` — порядок и связность этапов:** полный, пустой draft → корректный skip ветки, финал = draft.
- **Гвард пустых фактов** (1059–1083): ретрай + честный fail-fast, семантика верная.
- **`_filter_low_reliability_facts`** (1131): пороги и removal согласованы, аудит-бэкап пишется.
- **`_call_agent`:** HEART_MODEL_CHAIN (переключение по API-ошибке и пустому/короткому ответу, деградация до DeepSeek Pro), двойной ретрай по `finish_reason=length` с ростом потолка (×2 cap 24k для факт-агентов — свежая правка диффа, корректна), деградация external_reviewer на KIE-ошибке, подмена max_tokens↔max_completion_tokens, gemini-3-8 content-as-array — всё сходится.
- **`_chat_completion`:** retry только на транзиентных (429/timeout/conn/5xx), экспоненциальный backoff c jitter, капы через env — корректно; при недоступности классов openai — деградация без ретраев (не падает).
- **`_parse_json_response`:** снятие ограждений, прямой parse, сбалансированный скан `{}` со учётом строк/экранирования, висячие запятые — устойчивый, не бросает.
- **`_split_markdown_sections` / `_reassemble_sections`:** непрерывные срезы (join == text) — корректно.
- **`assert_invariants`:** H2-число, уровни, ±15%/20% по секциям, обрыв, утечки тегов — логика корректна.
- **`_programmatic_trim`:** обрезка по абзацам/предложениям с гарантией пунктуации — ок.
- **`_normalize_checklist_bullets`:** только для checklist-стиля, группы ≥2 строк — ложные срабатывания маловероятны.
- **Токен-аккаунтинг** в `_call_agent` — суммирование корректно.
- **`claim_ledger`:** `extract_claims` устойчив к 3 форматам входа; `build_claim_pack` — промоция pending по reliability/источнику согласована с гвардом `_filter_low_reliability_facts` (порядок «сначала claims, потом фильтр» — намеренный, коммит 64ae88f).
- **`registry` ↔ `prompts`:** покрытие агенты↔промпты 1:1, пересечений ключей между prompts_part1/2/3 нет (`dict.update` не теряет ничего).
- **Контракты вызовов:** `freshness.check_facts(facts)→(dict, list)`, `deliverables.build_content_package(article=…, llm_call=…)`, `cta_renderer.append_cta(article, direction, mode, topic_hint=…)` — сигнатуры совпадают с вызовами.
- **Свежий дифф (+266):** deliverables-этап, калькуляторный блок, redirect-фильтр источников, packager в registry — прочитаны построчно, кроме пунктов 2.1/2.6/2.7 проблем не найдено.

## 5. Что не проверено (выносится на следующие этапы)

- `norm_params.py` + 160 регрессионных тестов (этап 3) — включая функции, которые pipeline импортирует: `enforce_params`, `check_matrix_compliance`, `fix_matrix_violations`, `fix_markdown_tables`, `fix_table_of_contents`, `get_norm_params`, `check_draft`, `lint_form_deadline`, `lint_year_rules`, `detect_truncation`.
- `rag.py` (query_knowledge/format_rag_context, фильтр valid_until) и Qdrant-контракты (этап 4).
- `factcheck.py` (extract_claims/match_claims_to_facts/hedge_*/validate_law_references/check_future_laws/domain_authority — 1 286 строк, вызывается из 5 шагов), `jev_client`, `searxng/vane`, `freshness` внутрянка (этап 6).
- `deliverables.py`/`cta_renderer.py`/`math_validator.py`/`contradiction_checker.py` внутрянка (этапы 5–6).
