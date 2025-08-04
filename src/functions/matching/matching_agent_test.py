from src.functions.matching.matching_agent import MatchingAgent


def test_matching_agent():
    agent = MatchingAgent(
        keywords_path="src/functions/matching/test_keywords.csv",
        document_path="data/earnings_calls/ex1.xml"
    )

    matches = agent.cos_similarity(match_type="hybrid", exclude_duplicates=True)
    matches.export_to_json()
    # print(matches)

if __name__ == "__main__":
    test_matching_agent()
