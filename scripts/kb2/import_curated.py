"""
Импорт курируемой подборки в новую структуру knowledge_base/.

Копирует (не перемещает) файлы из исходных папок на Desktop в дерево
knowledge_base/. Список — результат анализа и утверждённой подборки.

Запуск:
  python scripts/kb2/import_curated.py          # копировать
  python scripts/kb2/import_curated.py --check  # только проверить наличие

Исходники не изменяются. Повторный запуск безопасен (перезапишет копию).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
KB_ROOT = PROJECT / "knowledge_base"

DESKTOP = Path(r"C:\Users\Анна\Desktop\сервер VPS")
SRC_COPYWRITER = DESKTOP / "Базы знаний" / "Я Копирайтер"
SRC_MARKETING = DESKTOP / "Базы знаний" / "Маркетинг"
SRC_METHODOLOGY = DESKTOP / "Базы знаний" / "Методологии"
SRC_PAYBACK = DESKTOP / "Базы знаний" / "Проверка на окупаемость"
SRC_FIN = DESKTOP / "База знаний Фин"


def _G(src_dir: Path, pattern: str, dst_rel: str, new_name: str | None = None) -> tuple[Path, str, str | None]:
    """Glob-резолвер для файлов с хрупкими именами (кавычки, смешанная раскладка, дубли)."""
    matches = sorted(src_dir.glob(pattern)) if src_dir.exists() else []
    if not matches:
        # sentinel: несуществующий путь с понятным именем для отчёта
        return (src_dir / pattern, dst_rel, new_name)
    return (matches[0], dst_rel, new_name)

# (источник, назначение) — относительные пути от корня соответствующей папки.
# Имя назначения можно поменять (очистка мусорных имён), источник — как на диске.
CURATED: list[tuple[Path, str, str]] = [
    # ── craft/writing (Писатель → heart) ───────────────────────────
    (SRC_COPYWRITER / "Писатель" / "Ilyahov_Yasno-ponyatno-Kak-donosit-mysli-i-ubezhdat-lyudey-s-pomoshchyu-slov.3sxfWA.711941.fb2", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Sarycheva_Pishi-sokrashchay-2025-Kak-sozdavat-silnyy-tekst.767633.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Zinser_Kak-pisat-horosho-Klassicheskoe-rukovodstvo-po-sozdaniyu-nehudozhestvennyh-tekstov.zvFEYw.353577.fb2", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Gal_Slovo-zhivoe-i-mertvoe.Zep2GA.678106.fb2", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Klark_50-priemov-pisma-ot-Roya-Pitera-Klarka.E0_nbg.680411.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Shugerman_Kak-sozdat-krutoy-reklamnyy-tekst.92n2Ew.574046.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Shugerman_Iskusstvo-sozdaniya-reklamnyh-poslaniy.fmsd6w.574056.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Aslanov_Kopirayting.Afy9hw.433671.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "copywriter_secrets.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Hendli_Pishut-vse-Kak-sozdavat-kontent-kotoryy-rabotaet.7ue8CA.612700.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Magic-Bullets.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Активный словарь русского языка. Том 2.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Russky_yazyk_i_kultura_rechi_-_Kortava_T_V_-_s.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Ogilvi_Ogilvi-o-reklame.XfIdsQ.350491.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "Berkshire Hathaway Letters to Shareholders - PDF Room.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "jeff-bezos-amazon-shareholder-letters-1997_2020.pdf", "craft/writing", None),
    (SRC_COPYWRITER / "Писатель" / "2025ltr.pdf", "craft/writing", "Berkshire_letter_2025.pdf"),

    # ── craft/editorial (Редактор → sheriff) ───────────────────────
    (SRC_COPYWRITER / "Редактор" / "Rozental_Klassicheskiy-spravochnik-po-russkomu-yazyku-Orfografiya-Punktuaciya-Orfograficheskiy-slovar-Propisnaya-ili-strochnaya-.787362.pdf", "craft/editorial", None),
    (SRC_COPYWRITER / "Редактор" / "Nepryahin_Anatomiya-zabluzhdeniy-Bolshaya-kniga-po-kriticheskomu-myshleniyu.Rke_yA.677109.fb2", "craft/editorial", None),
    (SRC_COPYWRITER / "Редактор" / "master-list-of-logical-fallacies.pdf", "craft/editorial", None),
    (SRC_COPYWRITER / "Редактор" / "Web-Content-Part-I-Newcastle-University-Web-Content-Standards.pdf", "craft/editorial", None),
    (SRC_COPYWRITER / "Редактор" / "Chicago-Manual-Style.pdf", "craft/editorial", None),
    (SRC_COPYWRITER / "Редактор" / "How_People_Read_on_the_Web.pdf", "craft/editorial", None),

    # ── craft/structure (Структурировщик → engineer) ───────────────
    (SRC_COPYWRITER / "Структурировщик" / "Minto_Princip-piramidy-Minto.609907.pdf", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "Makki_Istoriya-na-million-dollarov-Master-klass-dlya-scenaristov-pisateley-i-ne-tolko.4Fqwlw.656390.fb2", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "Makki_Dialog-Iskusstvo-slova-dlya-pisateley-scenaristov-i-dramaturgov.517706.pdf", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "Vogler_Puteshestvie-pisatelya-Mifologicheskie-struktury-v-literature-i-kino.404178.pdf", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "Kononov_Avtor-nozhnicy-bumaga.MFEyrQ.480918.fb2", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "Richard E. Mayer-Multimedia Learning-Cambridge University Press (2009).pdf", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "dokumen.pub_the-handbook-of-creative-writing-9780748689774.pdf", "craft/structure", None),
    (SRC_COPYWRITER / "Структурировщик" / "12-primerov-chek-listov-sokpro.ru_.pdf", "craft/structure", None),
    (SRC_FIN / "Поведенческие привычки" / "Kaneman_Dumay-medlenno-reshay-bystro.tbsahw.385321.fb2", "craft/structure", "Kaneman_Dumay-medlenno-reshay-bystro.fb2"),

    # ── craft/seo (Booster) ─────────────────────────────────────────
    (SRC_COPYWRITER / "SEO-оптимизатор" / "searchqualityevaluatorguidelines-09112025.pdf.pdf", "craft/seo", "Google_Search_Quality_Evaluator_Guidelines_2025-11.pdf"),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "hobo-strategic-seo-2025.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "2026 Go-To Manual for GEO.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "How GEN AI is Reshaping Search Engine by Aloha 2025.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "seo_2026.pdf", "craft/seo", "SEO_2026.pdf"),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Generative Engine Optimization AB testing.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Counterintuitive-SEO-0.1.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "SMXL-Milan-Schema-Markup-Masterclass-2.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "seocoach-structured-data-rich-snippets.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Definitive-Guide-to-Healthcare-Structured-Data-in-SEO.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Guidelines20for20publishing20structured20data_V2.0_20210416.pdf", "craft/seo", "Google_Structured_Data_Guidelines_V2.pdf"),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "guide-to-core-web-vitals.pdf", "craft/seo", None),
    _G(SRC_COPYWRITER / "SEO-оптимизатор", "core-web-vitals-simplified*", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "opengraph.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "headings.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "The-Holy-Grail-of-SEO_WordLift.pdf", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Запросы.txt", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Оптимизация1.txt", "craft/seo", None),
    (SRC_COPYWRITER / "SEO-оптимизатор" / "Разметка.txt", "craft/seo", None),

    # ── craft/visual (Artist) ───────────────────────────────────────
    (SRC_COPYWRITER / "Визуализатор" / "Thinking_with_Type.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "the-creatives-guide-to-branding.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "GL_BrandGuide.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "Josef+Albers-+Interaction+of+Color.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "AssociateBrand_VISID_200203.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "CPL_GraphicDesign_StyleGuide_042621_updated.pdf", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "GPT image.txt", "craft/visual", None),
    (SRC_COPYWRITER / "Визуализатор" / "НЕгатив промт.txt", "craft/visual", None),

    # ── business/marketing ──────────────────────────────────────────
    (SRC_MARKETING / "Kotler_kratkiy_2007.pdf", "business/marketing", None),
    (SRC_MARKETING / "Rays_Pozicionirovanie-Bitva-za-uznavaemost.453787.pdf", "business/marketing", None),
    (SRC_MARKETING / "Traut_Novoe-pozicionirovanie.825069.pdf", "business/marketing", None),
    (SRC_MARKETING / "Purple_cow.pdf", "business/marketing", None),
    (SRC_MARKETING / "Mann_Marketing-na-100-remiks.vmvFJQ.323854.pdf", "business/marketing", None),
    (SRC_MARKETING / "mann_i._-_bez_byudgeta._73_effektivnih_priema_marketinga.pdf", "business/marketing", None),
    (SRC_MARKETING / "Kak_rastut_brendy_O_chyom_ne_znayut_marketologi.pdf", "business/marketing", None),
    (SRC_MARKETING / "Epic content marketing _ Room.pdf", "business/marketing", "Epic_Content_Marketing_Pulizzi.pdf"),
    (SRC_MARKETING / "They_Ask_You_Answer_-_Marcus_Sheridan.pdf", "business/marketing", None),
    (SRC_MARKETING / "Content_Inc_Second_Edition_-_Joe_Pulizzi.pdf", "business/marketing", None),
    (SRC_MARKETING / "Метод StoryBrand.pdf", "business/marketing", None),
    _G(SRC_MARKETING, "Дамир-Халилов*", "business/marketing", None),
    (SRC_MARKETING / "Kalbah_Metod-Jobs-to-Be-Done-Proektirovanie-klientoorientirovannogo-produktalxD4yQ837657.pdf", "business/marketing", None),
    (SRC_MARKETING / "Berneys_Propaganda.616203.pdf", "business/marketing", None),
    (SRC_MARKETING / "CJM Как провести клиента.pdf", "business/marketing", None),
    (SRC_MARKETING / "ИССЛЕДОВАНИЕ ЦЕЛЕВОЙ АУДИТОРИИ.pdf", "business/marketing", None),
    (SRC_MARKETING / "Маркетинговая стратегия для самозанятых.pdf", "business/marketing", None),
    (SRC_MARKETING / "Практика применения системы показателей.pdf", "business/marketing", None),
    (SRC_MARKETING / "DIGITAL-МАРКЕТИНГ.pdf", "business/marketing", None),
    (SRC_COPYWRITER / "Исследователь" / "opora_rossii_ponyatie_reklamy_shatalina_01_08_2023.pdf", "business/marketing", None),

    # ── business/methodology ────────────────────────────────────────
    (SRC_METHODOLOGY / "Pine_Postroenie-biznes-modeley-Nastolnaya-kniga-stratega-i-novatora.446190.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Porter_Konkurentnaya-strategiya-Metodika-analiza-otrasley-i-konkurentov.weui4w.456668.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Moborn_Strategiya-golubogo-okeana-Kak-nayti-ili-sozdat-rynok-svobodnyy-ot-drugih-igrokov.7eodWw.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Goldratt_Cel_1_Cel-Process-nepreryvnogo-sovershenstvovaniya-.upl0MQ.268666.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Ris_Biznes-s-nulya.7QmsLQ.370498.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Blank_Chetyre-shaga-k-ozareniyu.363977.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "prakticheskie-aspekty-primeneniya-metodov-berezhlivogo-proizvodstva-v-ramkah-kontseptsii-teorii-ogranicheniy.pdf", "business/methodology", None),
    (SRC_METHODOLOGY / "Dalio_Principy-izmeneniya-mirovogo-poryadka.jNbDjA.692717.fb2"
        if (SRC_METHODOLOGY / "Dalio_Principy-izmeneniya-mirovogo-poryadka.jNbDjA.692717.fb2").exists()
        else SRC_COPYWRITER / "Исследователь" / "Dalio_Principy-izmeneniya-mirovogo-poryadka.jNbDjA.692717.fb2", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "Made_to_Stick_-_Chip_Heath.pdf", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "Obviously_Awesome_-_April_Dunford.pdf", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "Value_Proposition_Design_-_Alexander_Osterwalder.pdf", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "The_Mom_Test_-_Rob_Fitzpatrick.pdf", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "The_Minimalist_Entrepreneur_-_Sahil_Lavingia.pdf", "business/methodology", None),
    (SRC_FIN / "Малый бизнес" / "The_Secrets_of_Consulting_-_Gerald_M_Weinberg.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "Built_to_Sell_-_John_Warrillow.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "Company_of_One_-_Paul_Jarvis.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "Emyth_revisited_-_Michael_E_Gerber.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "The_Freelance_Way_-_Robert_Vlach.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "The_Win_Without_Pitching_Manifesto_-_Blair_Enns.pdf", "business/methodology", None),
    (SRC_FIN / "Фриланс" / "The_1-Page_Marketing_Plan__Get_New_Custome_-_Allan_Dib.pdf", "business/methodology", None),
    (SRC_FIN / "Переговоры" / "Never_split_the_difference_-_Chris_Voss.pdf", "business/methodology", None),
    (SRC_FIN / "Переговоры" / "Getting_to_Yes__Negotiating_Agreement_With_-_Roger_Fisher.pdf", "business/methodology", None),
    (SRC_FIN / "Переговоры" / "Crucial_Confrontations_-_Kerry_Patterson.pdf", "business/methodology", None),
    (SRC_FIN / "Поведенческие привычки" / "Nudge_The_Final_Edition_-_Richard_Thaler.pdf", "business/methodology", None),
    (SRC_FIN / "Поведенческие привычки" / "Scarcity_The_New_Science_of_Having_Less_and_How_it_Defines_our_Lives_-_Sendhil_Mullainathan.pdf", "business/methodology", None),
    (SRC_FIN / "Карьера" / "The_First_90_Days_Updated_and_Expanded_-_Michael_D_Watkins.pdf", "business/methodology", None),
    (SRC_FIN / "Карьера" / "_OceanofPDF.com_Range_How_Generalists_Triumph_in_a_Specialized_World_-_David_Epstein.pdf", "business/methodology", "Range_David_Epstein.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "Batyrev_45-tatuirovok-menedzhera.GynyqQ.352047.fb2", "business/methodology", None),
    (SRC_COPYWRITER / "Исследователь" / "Легко не будет.pdf", "business/methodology", None),

    # ── business/unit_economics ─────────────────────────────────────
    (SRC_MARKETING / "gajd.-kak-rasschitat-yunit-ekonomiku-dlya-wildberries-yandeks-marketa-i-ozon.pdf", "business/unit_economics", None),
    (SRC_MARKETING / "ЮНИТ экономика построение и расчет.pdf", "business/unit_economics", None),
    (SRC_MARKETING / "Prezentaciya_Master_klass_po_yunit_ekonomike.pdf", "business/unit_economics", None),
    (SRC_MARKETING / "Юнит-экономика_ новый подход к оценке эффективности предпринимательских проектов_.pdf", "business/unit_economics", None),
    _G(SRC_MARKETING, "ekonomika_i_finansy*", "business/unit_economics", None),
    _G(SRC_PAYBACK, "complete_business_metrics_guide.pdf", "business/unit_economics", None),
    (SRC_PAYBACK / "upravlenie-denezhnymi-potokami.pdf", "business/unit_economics", None),

    # ── business/finance ────────────────────────────────────────────
    (SRC_PAYBACK / "Фин_учет_Юдина.pdf", "business/finance", None),
    (SRC_PAYBACK / "Инвестиционный_анализ_-_уч._пособие.pdf", "business/finance", None),
    (SRC_FIN / "Финансовая опора" / "Hauzel_Psihologiya-deneg-Vechnye-uroki-bogatstva-zhadnosti-i-schastya.S2iX6w.665703.fb2", "business/finance", "Hauzel_Psihologiya-deneg.fb2"),
    (SRC_COPYWRITER / "Исследователь" / "ОАО.pdf", "business/finance", None),
    (SRC_COPYWRITER / "Исследователь" / "ООО.pdf", "business/finance", None),

    # ── business/reports (датированные, устаревающие) ───────────────
    (SRC_METHODOLOGY / "malyy-i-sredniy-biznes-v-rossii-v-2025-godu-i-prognoz-na-2026-2027-gody-v-logike-jtbd.pdf", "business/reports", "МСП_Россия_2025-2027_JTBD.pdf"),
    (SRC_MARKETING / "Бизнес-тренды 2026. Обзор патентного бюро Железно (общее).pdf", "business/reports", None),
    (SRC_MARKETING / "Тренды 2026 _ Студия Чижова.pdf", "business/reports", None),
    (SRC_MARKETING / "issledovanie-trendov.pdf", "business/reports", None),
    (SRC_MARKETING / "mezhdunarodnye-czifrovye-trendy-2026.pdf", "business/reports", None),
    (SRC_MARKETING / "trendy-v-podpisnoy-biznes-modeli.pdf", "business/reports", None),
    (SRC_MARKETING / "13022026-t-business-and-data-insight-niche-marketplaces-will-reach-990-billion-rubles-in-2025-doc.pdf", "business/reports", "T-Бизнес_маркетплейсы_2025.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "the-state-of-fashion-2026-vf.pdf", "business/reports", None),
    (SRC_PAYBACK / "izmeneniya-po-ved-2026.pdf", "business/reports", "ВЭД_изменения_2026.pdf"),
    (SRC_PAYBACK / "Артур Леер_Ассоц-я экспортеров и импортеров.pdf", "business/reports", None),

    # ── business/reference (справочники с датами) ───────────────────
    (SRC_COPYWRITER / "Исследователь" / "Таблица с пониженными ставками УСН по регионам на 2026 год.pdf", "business/reference", None),
    (SRC_COPYWRITER / "Исследователь" / "Osnovnye_zmeneniya_2026_godu.pdf", "business/reference", "Основные_изменения_2026.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "proizvodstvennyj-kalendar-2026.pdf", "business/reference", None),
    (SRC_COPYWRITER / "Исследователь" / "МРОТ по регионам.docx", "business/reference", None),
    (SRC_COPYWRITER / "Исследователь" / "31.05.2024_Uproshchennyy_uchet.pdf", "business/reference", None),
    (SRC_COPYWRITER / "Исследователь" / "ok-029-2014.pdf", "business/reference", "ОКВЭД_2.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "okved.pdf", "business/reference", "ОКВЭД_пояснения.pdf"),

    # ── legislation (что уже есть; заменится свежими редакциями) ────
    (SRC_COPYWRITER / "Исследователь" / "nkodeksrf.pdf", "legislation", "НК_РФ.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "Grazhdanskij-kodeks-Rossijskoj-Federacii-GK-RF-1.pdf", "legislation", "ГК_РФ_часть_1.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "Кодекс Российской Федерации об административных правонарушениях.pdf", "legislation", "КоАП_РФ.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "115 ФЗ.pdf", "legislation", "ФЗ-115.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "38-FZ-2024.pdf", "legislation", "ФЗ-38_о_рекламе.pdf"),
    (SRC_COPYWRITER / "Исследователь" / "11.07a40-185223-2024_20250313_reshenija_i_postanovlenija.pdf", "legislation", "Судебная_практика_2024.pdf"),

    # ── style_client ────────────────────────────────────────────────
    (SRC_COPYWRITER / "Писатель" / "Пример статьи1.txt", "style_client", None),
    (SRC_COPYWRITER / "Писатель" / "Пример статьи2.txt", "style_client", None),
    (SRC_COPYWRITER / "Писатель" / "Пример статьи3.txt", "style_client", None),
    (SRC_COPYWRITER / "Писатель" / "Ошибки.txt", "style_client", None),
]

# Обзор арбитражной практики — имя с битым байтом, копируем через glob-поиск префикса.
ARBITRAZH_PREFIX = "Обзор практики применения арбитражными судами"
ARBITRAZH_DST = "legislation/Обзор_арбитражной_практики_налоги.pdf"


def _copy_one(src: Path, dst_rel: str, new_name: str | None, check_only: bool) -> tuple[str, str, int]:
    if not src.exists():
        return ("MISS", str(src), 0)
    dst_dir = KB_ROOT / dst_rel
    dst = dst_dir / (new_name or src.name)
    if not check_only:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return ("OK", str(dst.relative_to(PROJECT)), src.stat().st_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только проверить источники, не копировать")
    args = ap.parse_args()

    ok = miss = 0
    total_bytes = 0
    missing: list[str] = []

    for src, dst_rel, new_name in CURATED:
        status, path, size = _copy_one(src, dst_rel, new_name, args.check)
        if status == "OK":
            ok += 1
            total_bytes += size
        else:
            miss += 1
            missing.append(path)

    # Битое имя обзора арбитражной практики — ищем по префиксу побайтово.
    src_dir = SRC_COPYWRITER / "Исследователь"
    arb = None
    if src_dir.exists():
        for p in src_dir.iterdir():
            try:
                name = p.name.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
            except Exception:
                name = ""
            if name.startswith(ARBITRAZH_PREFIX):
                arb = p
                break
    if arb is not None:
        status, path, size = _copy_one_named(arb, ARBITRAZH_DST, args.check)
        if status == "OK":
            ok += 1
            total_bytes += size
        else:
            miss += 1
            missing.append(path)
    else:
        miss += 1
        missing.append("Обзор арбитражной практики (не найден по префиксу)")

    print(f"Папка назначения: {KB_ROOT}")
    print(f"Скопировано/найдено: {ok} ({total_bytes / 1048576:.1f} МБ)")
    if missing:
        print(f"НЕ НАЙДЕНО: {miss}")
        for m in missing:
            print(f"  - {m}")
    return 0 if miss == 0 else 1


def _copy_one_named(src: Path, dst_rel_with_name: str, check_only: bool) -> tuple[str, str, int]:
    dst_rel, _, name = dst_rel_with_name.rpartition("/")
    return _copy_one(src, dst_rel, name, check_only)


if __name__ == "__main__":
    sys.exit(main())
