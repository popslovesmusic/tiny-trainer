import re

class BaseTokenizer:
    """Base class for tokenizers."""
    def __init__(self, patterns):
        self.patterns = patterns

    def _tokenize_text(self, text):
        """Tokenizes text based on a single regex pattern with capturing groups."""
        if not self.patterns:
            return text.split()
        
        # We join the patterns using a capturing group to get all matches.
        full_pattern = '|'.join(self.patterns)
        tokens = re.findall(f"({full_pattern})", text)
        return [token for token in tokens if token] # Filter out empty strings


class VSLTokenizer(BaseTokenizer):
    """
    Tokenizer for VSL-Core syntax.
    Handles alphanumeric identifiers, special VSL symbols, and punctuation.
    """
    def __init__(self):
        # We define a list of regex patterns, ordered from most specific to most general.
        vsl_patterns = [
            r'Tensor\[\d+×\d+\]',           # Matches tensors like Tensor[3×3]
            r'is_a_star',                   # Matches specific compound predicates
            r'is_a_machine',                # Matches specific compound predicates
            r'is_closed',                   # Matches specific compound predicates
            r'is_available',
            r'is_a_cube',
            r'is_a_novel',
            r'is_happy',
            r'is_correct',
            r'is_finished',
            r'is_difficult',
            r'is_empty',
            r'is_running',
            r'is_simple',
            r'is_made_of',
            r'is_at',
            r'is_cold',
            r'is_boiling',
            r'is_parked',
            r'is_important',
            r'is_on',
            r'is_red',
            r'is_loud',
            r'is_green',
            r'is_new',
            r'is_not',
            r'is_beautiful',
            r'is_black',
            r'is_blue',
            r'is_brown',
            r'is_charging',
            r'is_crying',
            r'is_of',
            r'is_round',
            r'is_studying',
            r'is_tall',
            r'is_warm',
            r'is_yellow',
            r'is_sunny',
            r'is_wet',
            r'is_large',
            r'is_clean',
            r'is_closed',
            r'is_happy',
            r'is_a_form_of',
            r'is_an_adult',
            r'is_a_machine',
            r'is_a_novel',
            r'is_a_star',
            r'is_a_cube',
            r'@t',                          # Matches the temporal symbol
            r'∀|∃|¬|∧|∨|→|↔|□|◇',          # Matches single-character logical symbols
            r'\w+',                         # Matches any word (alphanumeric and underscores)
            r'[():,.]',                     # Matches punctuation
        ]
        super().__init__(vsl_patterns)

    def tokenize(self, text):
        return self._tokenize_text(text)