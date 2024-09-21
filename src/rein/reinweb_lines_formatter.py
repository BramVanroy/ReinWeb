import re

from datatrove.pipeline.formatters.base import BaseFormatter

from rein.utils import is_latin


REMOVE_LINES_RE_PATTERNS = (
    r"^(read )?more\s*.*$",
    r"^(lees )?(meer|verder|ook)\s*[^\w\s]*",  # Any line starting with, e.g. `Lees meer »`
    r"^(direct|onmiddellijk|spring|nu)?\s+naar inhoud$",
    r"^snel\s*naar",  # Any line starting with
    r"^(tweet|like|vind leuk|deel)$",
    r"^deel (op|met)",  # Any line starting with
    r"^share",  # Any line starting with (we do not expect Dutch lines starting with `share`)
    r"^(print|druk)( (af|uit))?$",
    r"^(\d+|geen)\s*(comments|views|commentaren|reacties|connecties|bijdragen|likes|tweets|retweets|shares|volgers|followers|vind-ik-leuks|likes|sterren|stars)$",
    r"^(comments|views|commentaren|reacties|connecties|bijdragen|likes|tweets|retweets|shares|volgers|followers|vind-ik-leuks|likes|sterren|stars)[\s:]*(\d+|geen)$"
    r"^gebruik (van )?cookies$",
    r"^cookies$",
    r"^cookies (aanvaarden|weigeren)$",
    r"^(cookie|privacy)beleid$",
    r"^(privacy|gebruiks)\s*voorwaarden$",
    r"^(accept|decline|accepteren|weigeren)$",
    r"^(aan|uit)$",
    r"^sitemap$",
    r"^nieuwsbrief",  # Any line starting with
    r"^contactgegevens",  # Any line starting with
    r"^contact(eer)?\s\w*",  # Any line starting with
    r"^home\s*[^\w]",  # Any line starting with, e.g. breadcrumbs "Home > Dominicaanse Republiek > Bachata dansreis"
    r"^(quick )?view$",
    r"^(search|zoek|zoeken|hidden|(zoek|search) in (\w+)|\w+ zoekopdracht(en)? \w+)$",
)


class ReinwebLinesFilter(BaseFormatter):
    """
    Filter out lines that are not part of the main content of the page, as well as
    lines that are not in Latin script for more than a given threshold.
    Optionally remove empty lines.
    """

    name = " ⚞ ReinWeb Lines Filter"

    def __init__(
        self,
        remove_regex_matches: tuple[str, ...] = REMOVE_LINES_RE_PATTERNS,
        re_flags: int = re.IGNORECASE,
        min_latin_token_ratio: float = 0.89,
        keep_empty_lines: bool = True,
    ):
        super().__init__()
        self.remove_lines_re = (
            re.compile(r"|".join(REMOVE_LINES_RE_PATTERNS), flags=re_flags) if remove_regex_matches else None
        )
        self.min_latin_token_ratio = min_latin_token_ratio
        self.keep_empty_lines = keep_empty_lines

    def format(self, text: str) -> str:
        keep = []
        for line in text.splitlines(keepends=True):
            if not line.strip():
                if self.keep_empty_lines:
                    keep.append(line)
                continue

            # Only alphabetical chars, any script
            alpha_chs = [c for c in line if c.isalpha()]
            # If the line consists of only non-alpha characters (e.g. `2024`) then that's okay
            if len(alpha_chs) > 0:
                # Remove lines where the ratio of non-latin chars (e.g. Chinese, Arabic, etc.) is higher
                # than the threshold
                rw_latin_ch_ratio = len([c for c in alpha_chs if is_latin(c)]) / len(alpha_chs)
                if rw_latin_ch_ratio < self.min_latin_token_ratio:
                    continue

            if self.remove_lines_re and self.remove_lines_re.search(line):
                continue

            keep.append(line)

        return "".join(keep)
