import re
import uuid
from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.utils.lid import FT176LID, GlotLID

from rein.utils import is_latin


REMOVE_LINES_RE_PATTERNS = (
    r"^(?:read )?more(?:\W|$)",
    r"^(?:lees )?(?:meer|verder|ook)(?:\W|$)",
    r"^(?:direct|onmiddellijk|spring|nu)?\s+naar inhoud",
    r"^(?:jump )?to content",
    r"^pagina (?:niet )?gevonden",
    r"^terug\s*naar\s*boven",
    r"^back\s*to\s*top",
    r"^snel\s*naar",
    r"^(?:tweet|like|vind( ik)? leuk|share)(?:\W|$)",
    r"^volg (?:ons|me|mij|onze|mijn)(?:\W|$)",
    r"^(?:deel|share) (?:op|met|dit)\W",
    r"^(?:abonneer|subscribe)(?: (?:je|u|jezelf|uzelf))? (?:op|met|dit)\W",
    r"^schrijf(?: (?:je|u|jezelf|uzelf))? in(?:\W|$)",
    r"(?:pagina|page)\s*\d+\s*(?:van|\W+)\s*\d+(?:\W|$)",
    r"^(?:print|druk)(?: (?:af|uit))?$",
    r"^(?:\d+|geen|nul|zero)\s*(?:comments|views|kudos|commentaren|reacties|connecties|bijdragen|likes|tweets|retweets|shares|volgers|followers|vind-ik-leuks|likes|sterren|stars)",
    r"^(?:comments|views|kudos|commentaren|reacties|connecties|bijdragen|likes|tweets|retweets|shares|volgers|followers|vind-ik-leuks|likes|sterren|stars)\W*(?:\d+|geen|nul|zero)",
    r"^gebruik (?:van )?cookies(?:\W|$)",
    r"^cookies$",
    r"^disclaimer",
    r"^cookies\s*(?:aanvaarden|weigeren|op)(?:\W|$)",
    r"^(?:cookie|privacy)\s*beleid",
    r"^(?:cookie|privacy)\s*statement",
    r"^(?:cookie|privacy)\s*verklaring",
    r"^(?:privacy|gebruiks|algemene)\s*voorwaarden",
    r"^(?:accept|decline|accepteren|weigeren)$",
    r"(?:\W|^)(?:maakt|maken) gebruik van cookies",
    r"(?:\W|^)(?:gebruikt|gebruiken) cookies",
    r"^(?:aan|uit|ja|nee|ok|oké|okay|yes|no)$",
    r"^sitemap$",
    r"^(?:nieuwsbrief|news\s*letter)(?:\W|$)",
    r"^contactgegevens",
    r"^(?:ons )?contact(?:eer|eren)?(?:\W|$)",
    r"^contacteer (?:ons|mij|me)",
    r"^contactformulier$",
    r"^neem contact(?:\W|$)",
    r"^(?:quick\s*)?view",
    r"^(?:search|hidden)(?:\W|$)",
    r"^(?:\d+|geen|nul|zero)\s*(?:zoek)?(?:resultaten|opdrachten)(?:\W|$)",
    r"^(?:klik hier|click here)(?:\W|$)",
    r"^(?:gepubliceerd|geschreven|gepost)\s*(?:op|door|voor|in)",
    r"(?:\W|^)tel\.?(?:efoon)?(?:nummer)?\s*:?\s*\d+",
    r"(?:\W|^)e?-?mail\s*(?:adress?)?\s*:?\s*[\w\.-]+@[\w\.-]+\.\w+",
    r"(?:\W|^)fax\s*:?\s*\d+",
    r"^(?:volgend|vorig)e?\s*(?:bericht|pagina|post|comment|reactie|artikel|nieuws|blog|entry|item)",
    r"^(?:next|previous)\s*(?:post|page|comment|reaction|article|news|blog|entry|item)",
    r"^(?:uw|je|jouw) (?:web[\s-]]*)?browser (?:ondersteunt|accepteert|heeft|is|gebruikt)(?:\W|$)",
    r"^fotograaf:",
)


class ReinwebLinesFilter(BaseFormatter):
    """
    Filter out lines that are not part of the main content of the page, as well as
    lines that are not in Latin script for more than a given threshold.
    Optionally remove empty lines.
    """

    name = "🌊 ReinWeb Lines Filter"

    def __init__(
        self,
        remove_regex_matches: tuple[str, ...] = REMOVE_LINES_RE_PATTERNS,
        re_flags: int = re.IGNORECASE,
        min_latin_token_ratio: float = 0.89,
        keep_empty_lines: bool = True,
        languages: list[str] | str | None | Literal[False] = None,
        language_threshold: float = 0.65,
        backend: Literal["ft176", "glotlid"] = "ft176",
    ):
        super().__init__()
        re_str = "|".join([f"(?:{pat})" for pat in remove_regex_matches]) if remove_regex_matches else None
        self.remove_lines_re = re.compile(re_str, flags=re_flags) if remove_regex_matches else None
        self.min_latin_token_ratio = min_latin_token_ratio
        self.keep_empty_lines = keep_empty_lines

        self.language_threshold = language_threshold
        if isinstance(languages, str):
            languages = list(languages)
        self.languages = languages
        self.backend = backend
        if self.languages is not False:
            self.model = FT176LID(languages) if backend == "ft176" else GlotLID(languages)
        else:
            self.model = None

    def format(self, text: str) -> str:
        keep = []
        for line in text.splitlines(keepends=True):
            if not line.strip():
                if self.keep_empty_lines:
                    keep.append(line)
                continue

            if not self.filter_non_latin(line):
                continue

            if not self.filter_regex_matches(line):
                continue

            if not self.filter_lang:
                continue

            keep.append(line)

        return "".join(keep)

    def filter_non_latin(self, line: str) -> bool:
        # Only alphabetical chars, any script
        alpha_chs = [c for c in line if c.isalpha()]
        # If the line consists of only non-alpha characters (e.g. `2024`) then that's okay
        if len(alpha_chs) > 0:
            # Remove lines where the ratio of non-latin chars (e.g. Chinese, Arabic, etc.) is higher
            # than the threshold
            rw_latin_ch_ratio = len([c for c in alpha_chs if is_latin(c)]) / len(alpha_chs)
            if rw_latin_ch_ratio < self.min_latin_token_ratio:
                self.stat_update("rw_lines_filtered_non_latin")
                return False

        return True

    def filter_regex_matches(self, line: str) -> bool:
        if self.remove_lines_re and self.remove_lines_re.search(line.strip()):
            self.stat_update("rw_lines_filtered_regex")
            return False

        return True

    def filter_lang(self, line: str) -> bool:
        if self.model is not None:
            doc = Document(text=line, id=str(uuid.uuid4()))
            best_lang_pair, lang_pairs = self.model.predict(doc)
            lang, lang_score = best_lang_pair
            if self.backend == "glotlid":
                lang, _ = lang.split("_")

            if not (
                (self.languages is None and lang_score > self.language_threshold)
                or (self.languages and any(score > self.language_threshold for score in lang_pairs.values()))
            ):
                self.stat_update("rw_lines_filtered_language")
                return False

        return True
