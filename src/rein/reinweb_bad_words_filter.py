import re

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


REMOVE_BAD_WORDS_RE_PATTERNS = (
    # Adapted from https://en.wikipedia.org/wiki/Dutch_profanity
    "downie",
    "klere",
    "minkukel",
    r"(?:gad|get|god)ver",
    r"(?:god)?verdomme",
    "godskolere",
    "godverork",
    "kopvod",
    "anaalgeneraal",
    "bitch",
    r"fuck(?:ing)?",
    "klootzak",
    "klote",
    "kreng",
    "kringspiermusketier",
    "lul",
    "manwijf",
    "reetkever",
    "reetridder",
    "shit",
    "slet",
    "sodemieter",
    "stoephoer",
    "swaffel",
    "trut",
    "zeik",
    "bamivreter",
    "bosneger",
    "neger",
    "fransoos",
    "geitenneuker",
    "kaaskop",
    "koelie",
    "mocro",
    "moccro",  # https://www.ensie.nl/betekenis/mocro
    "nikker",
    "poepchinees",
    "spaghettivreter",
    "loempiavouwer",
    "spanjool",
    "spleetoog",
    "tatta",
    "tokkie",
    "zandneger",
    "halvezool",
    "klootviool",
    "oelewapper",
    "smeerlap",
    "wappie",
    # xxx (a.o. https://gitlab.com/yhavinga/c4nlpreproc/-/blob/master/clean/badwords_ennl.py?ref_type=heads)
    "anal",
    r"blow[-\s]*job",
    "cunt",
    "geil",
    "horny",
    r"(?:web[-\s]*)?cam[-\s]*se(?:x|ks)",
    r"se(?:x|ks)[-\s]*chat",
    r"se(?:x|ks)[-\s]*dat(?:ing|es?)",
    r"se(?:x|ks)[-\s]*contact(?:en)?",
    r"kut(?:je)?",
    "sex",  # Standaardnederlands = seks, maybe we catch some porn or socialmedia sites with this misspelling
    "porn",  # Standaardnederlands = porno
    "nigger",
    "nigga",
)


class ReinBadWordsFilter(BaseFilter):
    """
    Badwords filter from ReinWeb. Heaviliy inspired by the C4 Bad Words Filter.
    Args:
        TODO
    """

    name = "🌊 ReinWeb Bad Words Filter"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        remove_regex_matches: tuple[str, ...] = REMOVE_BAD_WORDS_RE_PATTERNS,
        re_flags: int = re.IGNORECASE,
        add_word_separation: bool = True,
        save_frequencies: bool = False,
    ):
        super().__init__(exclusion_writer)
        self.add_word_separation = add_word_separation
        bad_words_re_str = "|".join([f"(?:{pat})" for pat in remove_regex_matches])
        self.remove_docs_re = (
            re.compile(rf"(?:\W|^)(?:{bad_words_re_str})(?:\W|$)", flags=re.IGNORECASE)
            if add_word_separation
            else re.compile(bad_words_re_str, flags=re_flags)
        )
        self.save_frequencies = save_frequencies

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        badwords_found = (
            list(self.remove_docs_re.finditer(doc.text))
            if self.save_frequencies
            else self.remove_docs_re.search(doc.text)
        )

        if badwords_found:
            if self.save_frequencies:
                for badword in badwords_found:
                    badword = badword.group(0).strip()
                    self.stat_update(f"rw_badword_{badword}")
            return False, "rw_document_removed_with_badwords"

        return True
