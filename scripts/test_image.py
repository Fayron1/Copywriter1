import sys
import logging
from typing import Any, Optional
from openai import OpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_image")

def _extract_image_url(data: Any) -> Optional[str]:
    """Универсальный извлекатель URL из ответа API картинок."""
    if not data:
        return None
    if isinstance(data, str) and (data.startswith("http://") or data.startswith("https://")):
        return data
    if isinstance(data, list):
        for item in data:
            url = _extract_image_url(item)
            if url:
                return url
    if isinstance(data, dict):
        for key in ["url", "URL", "image", "image_url", "uri", "link"]:
            val = data.get(key)
            if val and isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return val
            if val and isinstance(val, dict):
                url = _extract_image_url(val)
                if url:
                    return url
        for key in ["data", "images", "output", "results", "response"]:
            val = data.get(key)
            if val:
                url = _extract_image_url(val)
                if url:
                    return url
        for key, val in data.items():
            if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                return val
            elif isinstance(val, (dict, list)):
                url = _extract_image_url(val)
                if url:
                    return url
    return None

def test_generation(api_key: str):
    client = OpenAI(api_key=api_key)
    
    # 1. Проверяем модель gpt-image-2 и пропорции 1536x768 через httpx
    logger.info("🚀 Пробуем сгенерировать тестовую картинку через gpt-image-2 (1536x768) с прямым HTTP POST...")
    try:
        import httpx
        base_url = str(client.base_url).rstrip('/')
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-image-2",
            "prompt": "Conceptual business cover, minimalist, neon green lights",
            "size": "1536x768",
            "n": 1
        }
        with httpx.Client(timeout=120.0) as http_client:
            resp = http_client.post(f"{base_url}/images/generations", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        logger.info(f"🔍 DEBUG: Сырой ответ API (gpt-image-2): {data}")
        cover_url = _extract_image_url(data)
        if cover_url:
            logger.info(f"✅ Успех gpt-image-2! Ссылка: {cover_url}")
        else:
            logger.error("❌ gpt-image-2 вернул ответ, но URL в нем не найден!")
    except Exception as e:
        logger.error(f"❌ Ошибка gpt-image-2 (прямой POST): {e}")

        
        # 2. Фолбэк тест: пробуем стандартную модель dall-e-3 через httpx
        logger.info("\n🔄 Пробуем запустить стандартный dall-e-3 (1024x1024) через прямой HTTP POST...")
        try:
            payload_dalle = {
                "model": "dall-e-3",
                "prompt": "Conceptual business cover, minimalist, neon green lights",
                "size": "1024x1024",
                "n": 1
            }
            with httpx.Client(timeout=120.0) as http_client:
                resp = http_client.post(f"{base_url}/images/generations", json=payload_dalle, headers=headers)
                resp.raise_for_status()
                data_dalle = resp.json()
            dalle_url = _extract_image_url(data_dalle)
            if dalle_url:
                logger.info(f"✅ Успех dall-e-3! Ссылка: {dalle_url}")
            else:
                logger.error("❌ dall-e-3 вернул ответ, но URL в нем не найден!")
            logger.info("ℹ️ Похоже, ваш API-ключ/прокси не поддерживает кастомную модель 'gpt-image-2' или размер '1536x768'. Рекомендуется использовать 'dall-e-3'.")
        except Exception as e2:
            logger.error(f"❌ Ошибка dall-e-3 (прямой POST): {e2}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python test_image.py <ВАШ_OPENAI_API_KEY>")
        sys.exit(1)
    
    test_generation(sys.argv[1])
