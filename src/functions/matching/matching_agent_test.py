from src.functions.matching.matching_agent import MatchingAgent
from src.abstract_classes.attribute import DocumentAttr

def test_matching_agent():

    test_document = DocumentAttr(
        "The company faces significant risk and uncertainty " +
        "in its market exposure. The impact of these factors " +
        "is being carefully monitored. There are risks in the " +
        "current market conditions that need attention."
    )

    print("1. Creating MatchingAgent instance...")
    agent = MatchingAgent(
        keywords_file="src/functions/matching/test_keywords.csv",
        document=test_document
    )

    matches = agent.cos_similarity(match_type="hybrid")
    print(matches)
    

if __name__ == "__main__":
    test_matching_agent()
