from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import Languages
from datatrove.utils.word_tokenizers import load_word_tokenizer

from rein.utils import is_latin, only_roman_chars


class ReinWebQualityFilter(BaseFilter):
    name = "🌊 ReinWeb Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        max_uppercase_token_ratio: float = 0.17,
        max_digit_token_ratio: float = 0.20,
        min_latin_token_ratio: float = 0.90,
        # Assuming 120 chars for a single sentence, we assume that the first letter must be uppercase (1/120)
        # In reality a single line often contains multiple sentences, so this value is a lower bound
        min_uppercase_char_ratio: float = 0.008,
        max_uppercase_char_ratio: float = 0.2,
        max_line_duplicates_ratio: float = 0.50,
        min_latin_char_ratio: float = 0.90,
        language: str = Languages.dutch,
    ):
        super().__init__(exclusion_writer)
        self.max_uppercase_token_ratio = max_uppercase_token_ratio
        self.max_digit_token_ratio = max_digit_token_ratio
        self.min_latin_token_ratio = min_latin_token_ratio
        self.min_uppercase_char_ratio = min_uppercase_char_ratio
        self.max_uppercase_char_ratio = max_uppercase_char_ratio
        self.max_line_duplicates_ratio = max_line_duplicates_ratio
        self.min_latin_char_ratio = min_latin_char_ratio
        self.tokenizer = load_word_tokenizer(language)

    def filter(self, doc) -> bool | tuple[bool, str]:
        tokens = self.tokenizer.word_tokenize(doc.text)
        num_tokens = len(tokens)

        if num_tokens == 0 or all(t == "" for t in tokens):
            return False, "rw_no_tokens"

        lines = doc.text.splitlines()
        unique_lines = set(lines)
        line_duplication_ratio = 1 - len(unique_lines) / len(lines)
        if line_duplication_ratio > self.max_line_duplicates_ratio:
            return False, "rw_above_line_duplication_ratio"

        # Tokens
        rw_uppercase_t_ratio = len([t for t in tokens if t.isupper()]) / num_tokens
        if rw_uppercase_t_ratio > self.max_uppercase_token_ratio:
            return False, "rw_above_uppercase_token_ratio"

        rw_digit_t_ratio = len([t for t in tokens if t.isdigit()]) / num_tokens
        if rw_digit_t_ratio > self.max_digit_token_ratio:
            return False, "rw_above_digit_token_ratio"

        rw_latin_t_ratio = len([t for t in tokens if only_roman_chars(t)]) / num_tokens
        if rw_latin_t_ratio < self.min_latin_token_ratio:
            return False, "rw_below_latin_token_ratio"

        # Alpabetic characters (any script)
        alpha_chs = [c for t in tokens for c in t if c.isalpha()]
        num_alpha_chs = len(alpha_chs)

        rw_uppercase_ch_ratio = len([c for c in alpha_chs if c.isupper()]) / num_alpha_chs
        if rw_uppercase_ch_ratio < self.min_uppercase_char_ratio:
            return False, "rw_below_uppercase_char_ratio"

        if rw_uppercase_ch_ratio > self.max_uppercase_char_ratio:
            return False, "rw_above_uppercase_char_ratio"

        rw_latin_ch_ratio = len([c for c in alpha_chs if is_latin(c)]) / num_alpha_chs
        if rw_latin_ch_ratio < self.min_latin_char_ratio:
            return False, "rw_below_latin_char_ratio"

        return True
