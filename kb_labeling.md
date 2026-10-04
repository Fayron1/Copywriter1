# Маркировка БЗ: что писать vs как писать (2026-10-04)

Главное правило системы: **материалы разделены на два слоя**, и агенты
получают только свой слой. Иначе Heart цитирует методологию как факт,
а fact_finder тащит Зинсера в фактчек.

## Два слоя

| Слой | kb_layer | Что это | Кто читает | Где лежит |
|---|---|---|---|---|
| **ЧТО** (факты) | `fact` | Доменные данные: нормы, тарифы, цифры, кейсы, методологии домена | fact_finder (+engineer/booster по папке) | legislation/, business/ |
| **КАК** (крафт) | `craft` | Ремесло: письмо, структура, редактура, стиль, убеждение | heart, sheriff, engineer, booster, artist — по папке | craft/, style_client/ |

Механизм: payload `agent_target` (список агентов) + `kb_layer` +
`domain` (папка-ключ) + `source_file` (полный путь) + `valid_until`
для датированных. Физически одна коллекция; при необходимости
выноса крафта в отдельную — RagConfig.collection уже per-agent.

## Правила папок лоадера (scripts/kb2/loader.py FOLDERS)

### ЧТО — факты (fact_finder)
| Папка | Тип | valid_until |
|---|---|---|
| legislation/ | law (P0) | — |
| business/marketplaces/ | snapshot | год файла → конец след. года |
| business/unit_economics/, finance/, reference/ | guide/reference | год → конец след. года |
| business/reports/ | report | год → конец след. года |
| business/marketing/paid_traffic_russia/ (рекурсивно) | snapshot | **90 дней** от снимка |
| business/marketing/crm_and_sales_automation/ (рекурсивно) | snapshot | **180 дней** |
| business/marketing/landing_pages_cro/ (рекурсивно) | snapshot | **180 дней** |
| business/marketing/b2b_email/ (рекурсивно) | etalon | — |
| business/marketing/b2b_content_cases_russia/ (рекурсивно) | case | **365 дней** |

### ЧТО — рамки домена (booster/engineer)
| Папка | Агенты |
|---|---|
| business/marketing/ (книги) | booster, engineer |
| business/methodology/ (ТРИЗ, Портер, Голдратт...) | engineer, heart |

### КАК — ремесло
| Папка | Агенты | Что внутри |
|---|---|---|
| craft/writing/ | heart, engineer | Зинсер, Ильяхов, Галь, Кларк, Пиши-сокращай, Хэндли, Шварц, Крон, [sliwbl] + словарь сочетаемости |
| craft/persuasion/ | heart, booster | Барден, Чалдини, Талер (Nudge), Мулланатан (Scarcity), barden_playbook |
| craft/structure/ | engineer | Минто, Канеман, Макки, Кононов |
| craft/editorial/ | sheriff | Розенталь, фаллоуции, How People Read |
| craft/seo/ | booster | GEO 2026, Google guidelines |
| style_client/ | heart, booster | примеры стиля |

## Границы (анти-путаница)

1. **Психология покупки — КАК**: Барден/Чалдини учат систему убеждать,
   не служат фактами в статьях. Когда статья О поведенческой экономике —
   факты берём из науки/исследований, а не из книги-популяризатора.
2. **Методология маркетинга — ЧТО-рамки**: JTBD, позиционирование,
   CRO-методика — booster/engineer используют как доменное знание
   для статей о маркетинге; fact_finder их не видит (это не факты).
3. **Снимки платформ — факты с TTL**: цифры тарифов умирают,
   valid_until ставится автоматически из даты снимка в имени файла.
4. **Кейсы — факты с ограничениями**: цифра только вместе с контекстом
   (карточка + снимок), лингер КЕЙС НЕ ГАРАНТИЯ на страже.
5. Дубли в Qdrant исключены: uuid5 от source_file+chunk+hash; перенесённый
   файл (Барден) перезаливается по новому пути, старые точки чистит prune.

## Порядок загрузки (2026-10-04)

1. Новые подпапки маркетинга (снимки + эталоны) — fact-слой.
2. craft/writing (дозагрузка недостающих) + craft/persuasion (новая).
3. Волны A–D книги (Канеман уже в structure; behavioural — persuasion).
4. Проверка: query по 3 маркерам из новых снимков через fact_finder.

## Решения по архитектурному консультанту (2026-10-04)

### Уже работало (консультант предлагал то, что есть)
- Гибрид dense+sparse с RRF — rag.py (2026-10-02)
- Firewall свежести valid_until (тройное should: актуальные/пустые/null) — fact_finder
- agent_target-фильтры = доступ агента только к своему срезу
- Автоснапшот перед --fresh, alias kb_v2, delete-by-source
- case_cards с limitations/checked_at, линтеры кейсов

### Внедрено в этот заход
- payload: kb_layer (fact/craft), domain-ключи, snapshot TTL → valid_until
- engineer подключён к фильтру свежести (протухшие тарифы не попадут в P&L)
- payload-индексы: agent_target, domain, source_type, valid_until, kb_layer
- Рекурсивная загрузка папок с longest-prefix правилами
- Политика: generated_article в evidence не индексируется (статьи системы
  не хранятся в knowledge_base/ — отдельная карантин-коллекция позже,
  если появится клиентский контур)

### Отложено (с причиной)
- 6 физических коллекций: изоляция уже через agent_target+kb_layer;
  вернуться при появлении клиентских namespace (kb_client_private)
- Evidence Cards: claim-ledger + claim-pack уже дают связку
  утверждение→источник→дата; эволюционируем существующее
- Граф связей: DIRECTION_BOOST_PREFIXES покрывает маршрутизацию;
  граф нужен при 10+ доменах
- tenant_id: клиентов пока нет; kb_update гейтит по chat_id
- Retrieval-QA сет (30–50 запросов): сделать после первой загрузки
  по 5 маркерам новых доменов
