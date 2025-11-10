from abc import ABC, abstractmethod
from typing import Any
import xml.etree.ElementTree as ET
import re
from .attribute import DocumentAttr


class AbstractDocumentLoader(ABC):
    """
    Abstract base class for loaders that convert various raw data sources
    into a structured DocumentAttr object.
    """

    @abstractmethod
    def load(self, file_path: str) -> DocumentAttr:
        """
        Load and structure a document from a given file path.
        
        Args:
            file_path (str): Path to the raw file (XML, JSON, TXT, etc.)
        
        Returns:
            DocumentAttr: A structured document representation.
        """
        pass


class XMLTranscriptLoader(AbstractDocumentLoader):
    """
    Loader for XML earnings call transcripts.
    Extracts presentation and Q&A sections and returns a DocumentAttr.
    """

    def _extract_presentation_section(self, xml_path: str) -> str:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        body_elem = root.find(".//Body")
        body = body_elem.text if body_elem is not None else None
        lines = body.splitlines() if body is not None else []

        in_presentation = False
        presentation_lines = []

        for line in lines:
            if re.search(r"^\s*Presentation\s*$", line):
                in_presentation = True
                continue
            if re.search(r"^\s*(Questions and Answers|Q&A|Operator)\s*$", line, re.IGNORECASE):
                break
            if in_presentation:
                presentation_lines.append(line)

        return "\n".join(presentation_lines).strip()

    def _extract_qa_section(self, xml_path: str) -> str:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        body_elem = root.find(".//Body")
        body = body_elem.text if body_elem is not None else None
        lines = body.splitlines() if body is not None else []

        in_qa = False
        qa_lines = []

        for line in lines:
            if not in_qa and re.search(r"^\s*(Questions and Answers|Q&A)\s*$", line, re.IGNORECASE):
                in_qa = True
                continue
            if in_qa:
                qa_lines.append(line)

        return "\n".join(qa_lines).strip()

    def _clean_spoken_content(self, raw_text: str) -> str:
        lines = raw_text.splitlines()
        cleaned_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.fullmatch(r'[-=]{5,}', line):
                i += 3  # skip separator, speaker, separator
                continue
            cleaned_lines.append(lines[i])
            i += 1

        return "\n".join(cleaned_lines).strip()

    def load(self, file_path: str) -> DocumentAttr:
        try:
            presentation_text = self._extract_presentation_section(file_path)
            qa_text = self._extract_qa_section(file_path)
            full_text = presentation_text + "\n\n" + qa_text
            cleaned_text = self._clean_spoken_content(full_text)

            return DocumentAttr(document=cleaned_text)

        except Exception as e:
            print(f"Error loading document: {e}")
            return DocumentAttr(document="")


class TXTDocumentLoader(AbstractDocumentLoader):
    """
    Loader for plain text files.
    Reads the file content and returns a DocumentAttr.
    """

    def load(self, file_path: str) -> DocumentAttr:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            
            # Here you could add preprocessing, e.g. cleaning
            # For now, just wrap the raw text
            return DocumentAttr(document=text)

        except Exception as e:
            print(f"Error loading TXT document: {e}")
            return DocumentAttr(document="")
        
