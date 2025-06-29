from src.functions.matching.matching_agent import MatchingAgent
from src.abstract_classes.attribute import DocumentAttr
import os

def test_matching_agent():

    test_document = DocumentAttr(
        "The company faces significant risk and uncertainty " +
        "in its market exposure. The impact of these factors " +
        "is being carefully monitored. There are risks in the " +
        "current market conditions that need attention."
    )

    print("1. Creating MatchingAgent instance...")
    agent = MatchingAgent(
        keywords_file="test_keywords.csv",
        document=test_document
    )

    # print("\n2. Loaded keywords:")
    # print(f"Number of keywords (including variations): {len(agent.keywords_list)}")
    # print("Sample keywords:", agent.keywords_list[:10])
    #
    print("\n3. Testing cosine similarity matching...")
    print("Document text:", test_document.text)
    result = agent.cos_similarity(match_type="single")
    print("Cosine similarity results:", result)
    #
    # print("\n4. Testing with custom threshold...")
    # result_strict = agent.cos_similarity(match_type="single", threshold=0.9)
    # print("Strict threshold results:", result_strict)
    #
    # # Clean up
    # if os.path.exists(test_keywords_file):
    #     os.remove(test_keywords_file)
    # print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_matching_agent()
