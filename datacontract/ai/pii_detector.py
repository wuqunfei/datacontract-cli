from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datacontract.config import Config


class PiiDetector(ABC):
    """Decides whether column names are likely to contain personally identifiable information."""

    def __init__(self, config: "Config | None" = None) -> None:
        self.config = config

    @abstractmethod
    def detect(self, column_names: list[str]) -> dict[str, bool]:
        """Return {column_name: is_pii} for a batch of column names from one table."""
