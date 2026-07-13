import re

ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")
ARABIC_TATWEEL = re.compile("ـ")
NON_ARABIC_CHARACTERS = re.compile(r"[^ء-غف-ي\s]")
MULTIPLE_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """
    Normalise un texte arabe pour rendre les comparaisons indépendantes de
    la vocalisation, des variantes orthographiques et de la ponctuation.
    """
    text = ARABIC_DIACRITICS.sub("", text)
    text = ARABIC_TATWEEL.sub("", text)
    text = re.sub("[إأآٱا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = NON_ARABIC_CHARACTERS.sub(" ", text)
    return MULTIPLE_WHITESPACE.sub(" ", text).strip()
