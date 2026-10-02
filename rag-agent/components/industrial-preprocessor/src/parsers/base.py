"""Abstract interface shared by all document parsers."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.document import Document


class BaseParser(ABC):
    """Convert an input path into the parser-independent Document model."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> Document:
        """Parse ``file_path`` and return a structured Document."""

        raise NotImplementedError
