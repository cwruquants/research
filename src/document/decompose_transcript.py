import xml.etree.ElementTree as ET
import re
from src.abstract_classes.attribute import DocumentAttr

__all__ = [
    "extract_presentation_section",
    "extract_qa_section",
    "clean_spoken_content"
]

def extract_presentation_section(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract the full transcript body
    body_elem = root.find(".//Body")
    body = body_elem.text if body_elem is not None else None
    lines = body.splitlines() if body is not None else []

    # Flags and buffers
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


def extract_qa_section(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract the full transcript body
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


def clean_spoken_content(raw_text: str) -> str:
    """
    Removes speaker names and dashed section separators from transcript sections.
    
    Args:
        raw_text (str): The raw extracted transcript section (e.g., from presentation or Q&A).
    
    Returns:
        str: Cleaned text with only spoken content.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip lines that are separator dashes (---- or ====)
        if re.fullmatch(r'[-=]{5,}', line):
            # Skip the next line too (speaker name), and next separator
            i += 3  # jump over separator, speaker, separator
            continue

        cleaned_lines.append(lines[i])
        i += 1

    return "\n".join(cleaned_lines).strip()


def load_document(file_path: str) -> DocumentAttr:
    """
    Load a sample XML earnings call transcript and extract its text content
    using the decompose_transcript functions.
    Returns a DocumentAttr object with the text.
    """
    try:
        # Extract presentation and Q&A sections
        presentation_text = extract_presentation_section(file_path)
        qa_text = extract_qa_section(file_path)
        
        # Combine sections
        full_text = presentation_text + "\n\n" + qa_text
        
        # Clean spoken content to remove speaker tags and separators
        cleaned_text = clean_spoken_content(full_text)
        
        return DocumentAttr(document=cleaned_text)
    except Exception as e:
        print(f"Error loading document: {e}")
        return DocumentAttr(document="")