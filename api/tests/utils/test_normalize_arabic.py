from app.utils.normalize_arabic import normalize_arabic


def test_normalize_arabic_removes_diacritics_and_normalizes_letters():
    assert normalize_arabic("إِنَّ هٰذِهِۦ سَبِيلَى") == "ان هذه سبيلي"


def test_normalize_arabic_normalizes_alif_wasla():
    assert normalize_arabic("ٱللَّهُ") == "الله"


def test_normalize_arabic_removes_tatweel_and_punctuation():
    assert normalize_arabic("قُـلْ، هُوَ اللَّهُ أَحَدٌ!") == "قل هو الله احد"


def test_normalize_arabic_removes_non_arabic_content_without_joining_words():
    assert normalize_arabic("قل [Whisper 123] هو الله") == "قل هو الله"


def test_normalize_arabic_collapses_whitespace_and_trims_result():
    assert normalize_arabic("  قل\n\tهو   الله  ") == "قل هو الله"


def test_normalize_arabic_returns_empty_string_without_arabic_letters():
    assert normalize_arabic("Whisper: 123!") == ""
