import pytest

from app.core.transcription_policy import is_confidently_non_arabic


@pytest.mark.parametrize(
    ("language", "language_probability", "arabic_probability"),
    [
        ("fr", 0.98, 0.0),
        ("en", 0.62, 0.003),
        ("fi", 0.37, 0.006),
    ],
)
def test_rejects_non_arabic_language_when_arabic_is_not_plausible(
    language,
    language_probability,
    arabic_probability,
):
    assert is_confidently_non_arabic(
        language,
        language_probability,
        arabic_probability,
    )


@pytest.mark.parametrize(
    ("language", "language_probability", "arabic_probability"),
    [
        ("ar", 0.50, 0.50),
        ("en", 0.55, 0.08),
        ("fr", 0.61, 0.10),
        (None, None, None),
    ],
)
def test_preserves_arabic_or_uncertain_audio_for_quran_decoding(
    language,
    language_probability,
    arabic_probability,
):
    assert not is_confidently_non_arabic(
        language,
        language_probability,
        arabic_probability,
    )
