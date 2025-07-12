from src.functions.matching.matching_agent import MatchingAgent
from src.abstract_classes.attribute import DocumentAttr
from src.functions.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content

def test_matching_agent():
    agent = MatchingAgent(
        keywords_file="src/functions/matching/test_keywords.csv",
        document=load_sample_document("data/earnings_calls/ex1.xml")
    )

    matches = agent.cos_similarity(match_type="word")
    print(matches)

def load_sample_document(file_path: str) -> DocumentAttr:
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

if __name__ == "__main__":
    test_matching_agent()
