from src.functions.matching.matching_agent import MatchingAgent
from src.functions.decompose_transcript import load_sample_document

def test_matching_agent():
    agent = MatchingAgent(
        keywords_file="src/functions/matching/test_keywords.csv",
        document=load_sample_document("data/earnings_calls/ex1.xml")
    )

    matches = agent.cos_similarity(match_type="hybrid", exclude_duplicates=True)
    print(matches)

if __name__ == "__main__":
    test_matching_agent()
