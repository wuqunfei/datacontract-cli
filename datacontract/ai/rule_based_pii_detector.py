import re

from datacontract.ai.pii_detector import PiiDetector

_PII_TERMS = [
    "email",
    "e_mail",
    "ssn",
    "social_security",
    "phone",
    "mobile",
    "address",
    "street",
    "city",
    "zip",
    "postal",
    "dob",
    "birth_date",
    "date_of_birth",
    "first_name",
    "last_name",
    "full_name",
    "firstname",
    "lastname",
    "passport",
    "credit_card",
    "card_number",
    "iban",
    "tax_id",
    "national_id",
    "driver_license",
]
_PII_TERM_SEGMENTS = [term.split("_") for term in _PII_TERMS]


class RuleBasedPiiDetector(PiiDetector):
    """Matches column names against a fixed dictionary of PII-indicating terms. No config, no network."""

    def detect(self, column_names: list[str]) -> dict[str, bool]:
        return {name: self._is_pii(name) for name in column_names}

    @staticmethod
    def _is_pii(column_name: str) -> bool:
        segments = re.sub(r"[^a-z0-9]+", "_", column_name.lower()).strip("_").split("_")
        return any(_contains_subsequence(segments, term_segments) for term_segments in _PII_TERM_SEGMENTS)


def _contains_subsequence(segments: list[str], term_segments: list[str]) -> bool:
    n = len(term_segments)
    return any(segments[i : i + n] == term_segments for i in range(len(segments) - n + 1))
