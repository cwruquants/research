from src.functions.matching.matching_agent import MatchingAgent

def test_matching_agent_batch_processing():
    agent = MatchingAgent(
        keywords_path="src/test/test_keywords.csv",
        cos_threshold=0.7,
    )

    results_by_file = agent.batch_processing(
        folder_path="data/earnings_calls/2016",
        matching_function="cosine",  # use "direct" for exact matching
        match_type="word",           # used only when matching_function="cosine"
        print_results=False,        # set True to print each result
        save_json=True,             # saves under results/exposure_run_<timestamp>/
        export_format="word",       # "word" or "sentence" export format
        output_root_dir="results",  # output directory
    )

    print(f"Processed {len(results_by_file)} files from data/earnings_calls/2016")


if __name__ == "__main__":
    test_matching_agent_batch_processing()
