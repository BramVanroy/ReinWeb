import unicodedata as ud


STOP_WORDS_DUTCH = [
    "de",
    "het",
    "een",
    "die",
    "dat",
    "en",
    "of",
    "maar",
    "ben",
    "bent",
    "is",
    "zijn",
    "was",
    "waren",
    "word",
    "wordt",
    "worden",
    "werd",
    "werden",
    "heb",
    "hebt",
    "heeft",
    "hebben",
    "had",
    "hadden",
    "van",
    "op",
    "naar",
    "toe",
    "uit",
    "in",
    "door",
    "over",
    "onder",
    "voor",
    "achter",
    "tegen",
    "tussen",
    "mijn",
    "jouw",
    "zijn",
    "haar",
    "ons",
    "jullie",
    "hun",
]


_LATIN_LETTERS = {}


# Taken from https://stackoverflow.com/a/3308844/1150683
def is_latin(unicode_chr: str) -> bool:
    try:
        return _LATIN_LETTERS[unicode_chr]
    except KeyError:
        try:
            return _LATIN_LETTERS.setdefault(unicode_chr, "LATIN" in ud.name(unicode_chr))
        except Exception:
            return False


def only_roman_chars(unicode_text: str) -> bool:
    return all(is_latin(uchr) for uchr in unicode_text if uchr.isalpha())


_FLOATABLE_STRINGS = {"nan", "inf", "-inf", "infinity", "-infinity"}


def is_number(text: str) -> bool:
    text = text.lower()
    if text in _FLOATABLE_STRINGS:
        return False  # Despite these being "floatable", we consider them as regular tokens
    try:
        float(text)
        return True
    except ValueError:
        return False
