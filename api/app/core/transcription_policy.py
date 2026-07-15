# ROLE
# ----
# Seuils conservateurs utilisés pour écarter un audio avant le matching coranique.

NON_ARABIC_LANGUAGE_MIN_PROBABILITY = 0.75
NON_ARABIC_MAX_ARABIC_PROBABILITY = 0.15
UNCERTAIN_NON_ARABIC_LANGUAGE_MIN_PROBABILITY = 0.35
VERY_LOW_ARABIC_PROBABILITY = 0.05

MIN_AVERAGE_LOG_PROBABILITY = -1.0
HIGH_TEMPERATURE = 0.8
HIGH_TEMPERATURE_MAX_LOG_PROBABILITY = -0.8
HIGH_COMPRESSION_RATIO = 2.4
HIGH_COMPRESSION_MAX_LOG_PROBABILITY = -0.7


def is_confidently_non_arabic(
    language: str | None,
    language_probability: float | None,
    arabic_probability: float | None,
) -> bool:
    """Rejette une langue étrangère forte ou un score arabe quasi nul.

    Le second cas couvre notamment musique et bruit, dont la langue gagnante
    reste souvent peu probable. Une probabilité arabe non négligeable conserve
    volontairement l'audio pour le décodage coranique.
    """
    if (
        language in (None, "ar")
        or language_probability is None
        or arabic_probability is None
    ):
        return False

    return (
        language_probability >= NON_ARABIC_LANGUAGE_MIN_PROBABILITY
        and arabic_probability < NON_ARABIC_MAX_ARABIC_PROBABILITY
    ) or (
        language_probability >= UNCERTAIN_NON_ARABIC_LANGUAGE_MIN_PROBABILITY
        and arabic_probability < VERY_LOW_ARABIC_PROBABILITY
    )
