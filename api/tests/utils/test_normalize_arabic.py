from app.utils.normalize_arabic import normalize_arabic


def test_normalize_arabic_removes_diacritics_and_normalizes_letters():
    assert normalize_arabic("إِنَّ هٰذِهِۦ سَبِيلَى") == "ان هذه سبيلي"


def test_normalize_arabic_normalizes_alif_wasla():
    assert normalize_arabic("ٱللَّهُ") == "الله"
