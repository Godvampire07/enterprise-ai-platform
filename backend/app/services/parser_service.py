from abc import ABC, abstractmethod
from typing import List, Dict, Any
import fitz  # PyMuPDF

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse file and return structured page objects:
        [
            {
                "page": 1,
                "text": "..."
            }
        ]
        """
        pass

class PDFParser(BaseParser):
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise RuntimeError(f"PyMuPDF failed to open file '{file_path}': {str(e)}")
        
        pages = []
        try:
            for i, page in enumerate(doc):
                text = page.get_text()
                pages.append({
                    "page": i + 1,
                    "text": text or ""
                })
        finally:
            doc.close()
        return pages

class ParserService:
    def __init__(self) -> None:
        self._parsers = {
            "pdf": PDFParser()
        }

    def parse_file(self, file_path: str, content_type: str) -> List[Dict[str, Any]]:
        content_type_lower = content_type.lower()
        file_path_lower = file_path.lower()

        if "pdf" in content_type_lower or file_path_lower.endswith(".pdf"):
            return self._parsers["pdf"].parse(file_path)
            
        raise ValueError(f"Unsupported content type or file extension: {content_type}")
