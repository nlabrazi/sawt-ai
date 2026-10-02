import pytest

from app.services.hadith_query import normalize_hadith_query


@pytest.mark.parametrize("query, expected", [
    ("Je cherche le hadith sur la colère", "la colère"),
    ("Trouve moi le ou les hadith qui parlent du mariage", "du mariage"),
    ("Trouve-moi les hadiths qui parlent du mariage.", "du mariage."),
    ("Peux-tu me trouver un ou plusieurs hadiths concernant le mariage ?", "le mariage ?"),
    ("Trouve-moi le ou les hadiths qui parlent du mariage sans divorce", "du mariage sans divorce"),
    ("Donnez moi hadith qui parle de la colère", "la colère"),
    ("Donnez-moi un hadith qui parle de la colère", "la colère"),
    ("S’il vous plaît, pouvez-vous me donner un hadith sur la colère", "la colère"),
    ("  JE CHERCHE LE HADITH SUR la colère  ", "la colère"),
    ("J’aimerais retrouver le hadith au sujet de la prière", "la prière"),
    ("Un hadith qui dit que nos actes comptent selon l'intention qu'on avait", "nos actes comptent selon l'intention qu'on avait"),
    ("Un hadith qui dit qu’il ne faut pas se mettre en colère", "il ne faut pas se mettre en colère"),
    ("Je cherche le hadith où le Prophète conseille de ne pas se mettre en colère", "le Prophète conseille de ne pas se mettre en colère"),
    ("Donnez-moi un hadith sur la colère sans violence envers les enfants", "la colère sans violence envers les enfants"),
    ("Un hadith qui parle d’un homme qui ne priait pas", "un homme qui ne priait pas"),
    ("Un hadith qui parle des parents", "des parents"),
    ("Un hadith qui parle du pardon", "du pardon"),
    ("Un hadith qui parle de لا تغضب", "لا تغضب"),
])
def test_removes_only_the_request_and_preserves_the_subject(query, expected):
    assert normalize_hadith_query(query) == expected
    assert normalize_hadith_query(expected) == expected


@pytest.mark.parametrize("query", [
    "Ne pas se mettre en colère",
    "Je ne cherche pas un hadith sur la colère",
    "Ne trouve pas le ou les hadiths qui parlent du mariage",
    "Trouve-moi le ou les hadiths qui ne parlent pas du mariage",
    "Je cherche un hadith qui ne parle pas de la colère",
    "Je cherche un hadith qui parle seulement de la colère",
    "Un homme dit : donnez moi un hadith sur la colère",
    "La différence entre un hadith sur la colère et un hadith sur le pardon",
    "Le hadith interdit-il la colère ?",
    "Un hadith qui parle de",
    "Un hadith sur ...",
    "Un hadith sur la",
    "Un hadith sur je cherche un hadith sur la colère",
    "La religion est la sincérité",
    "Le conseil de parler seulement pour dire du bien sinon se taire",
    "Je cherche le conseil de faciliter les choses et de ne pas faire fuir les gens",
    "من كان يؤمن بالله واليوم الآخر فليقل خيرا أو ليصمت",
    "",
])
def test_preserves_negations_meaningful_content_and_incomplete_or_unknown_requests(query):
    assert normalize_hadith_query(query) == query
