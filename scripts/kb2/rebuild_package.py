"""Досборка контент-пакета для готовой статьи (без перегенерации).

Запуск на VPS:
    cd /root/Copywriter1 && venv/bin/python scripts/kb2/rebuild_package.py /root/article15
"""
import sys, json, glob
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")

from agents.pipeline import Pipeline, PipelineState
from agents.deliverables import build_content_package, package_to_markdown


def main(base: str):
    art_dir = sorted(glob.glob(f"{base}/*/article.md"))[-1]
    article = Path(art_dir).read_text(encoding="utf-8")

    # claims/references из debug-выгрузки, если есть
    claims, refs = [], []
    dbg = Path(art_dir).parent / "pipeline_debug.json"
    if dbg.exists():
        try:
            d = json.loads(dbg.read_text(encoding="utf-8"))
            claims = d.get("claims") or []
            refs = d.get("references") or []
        except Exception:
            pass

    import os
    pipe = Pipeline(openai_api_key=os.getenv("OPENAI_API_KEY", ""))
    state = PipelineState()
    state.provider = "deepseek"
    state.topic = "Типичные ошибки при оформлении электронного трудового договора"
    state.direction = "трудовое"

    pkg = build_content_package(
        article=article, topic=state.topic, direction=state.direction,
        claims=claims, references=refs, article_year=2026,
        llm_call=lambda p: pipe._call_agent("packager", p, state=state, parse_json=False),
        cta_mode="self_promo")

    out = Path(art_dir).parent / "content_package.md"
    out.write_text(package_to_markdown(pkg), encoding="utf-8")
    print("saved:", out)
    print("faq:", len(pkg.get("faq", [])),
          "| tg:", len(pkg.get("telegram_posts", [])),
          "| meta:", len(pkg.get("meta_description", "")),
          "| announce:", "short_announcement" in pkg)
    print(json.dumps({k: v for k, v in pkg.items()
                      if k in ("seo_title", "meta_description", "short_announcement")},
                     ensure_ascii=False, indent=1)[:700])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/article15")
