# 🔤 Normalisation du texte arabe
import re

# Liste des diacritiques à supprimer (tashkeel)
ARABIC_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")

def normalize_arabic(text):
    """
    Normalise le texte arabe :
    - Supprime les diacritiques (tashkeel)
    - Remplace certaines lettres par une forme standard
    """
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[إأآٱا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text
