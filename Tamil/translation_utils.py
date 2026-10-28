# translation_utils.py
import time
import re
from deep_translator import GoogleTranslator, MyMemoryTranslator

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    # Filter out English letters (tii, ruu, etc.) and retain ONLY Tamil script characters
    tamil_only = "".join([c for c in text if '\u0B80' <= c <= '\u0BFF' or c.isspace()])
    return re.sub(r'\s+', ' ', tamil_only).strip()


def translate_tamil_to_kannada(text_str: str) -> str:
    """
    Translates Tamil text to meaningful Kannada sentences.
    Uses Google Translate as the primary engine and seamlessly falls back 
    to MyMemory Translation if a network or server error occurs.
    """
    cleaned_text = clean_ocr_text(text_str)
    if not cleaned_text:
        return ""

    # Strategy 1: Primary - Google Translator
    try:
        time.sleep(0.3)
        translator = GoogleTranslator(source='ta', target='kn')
        result = translator.translate(cleaned_text)
        if result and result.strip() != cleaned_text.strip():
            return result
    except Exception:
        pass  # Fall through to secondary engine if Google fails

    # Strategy 2: Secondary Engine - MyMemory Translator (Free & Highly Reliable for Indic Languages)
    try:
        time.sleep(0.3)
        mymemory_translator = MyMemoryTranslator(source='ta-IN', target='kn-IN')
        result = mymemory_translator.translate(cleaned_text)
        if result and not result.startswith("MYMEMORY WARNING"):
            return result
    except Exception:
        pass

    # Strategy 3: Retry MyMemory with generic language codes ('ta', 'kn')
    try:
        time.sleep(0.3)
        mymemory_generic = MyMemoryTranslator(source='ta', target='kn')
        result = mymemory_generic.translate(cleaned_text)
        if result:
            return result
    except Exception:
        pass

    return "[Translation Error: Service temporarily unreachable. Please check network.]"


def translate_batch_tamil_to_kannada(text_list: list[str]) -> list[str]:
    """
    Translates a list of Tamil lines into Kannada using multi-engine routing.
    """
    if not text_list:
        return []

    return [translate_tamil_to_kannada(line) for line in text_list]