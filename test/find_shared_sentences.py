"""
Script to find shared sentences between keyword sets for a set earnings calls.
"""
from pathlib import Path
from src.analysis.comparison.result_analysis import find_shared_sentences

results_dir = "results/2016" # the directory containing the results of the earnings calls
keyword_names = ["Activeness.csv", "risk.csv"] # the keyword sets to compare

print(f"Finding shared sentences in {results_dir}")
print(f"Keyword sets: {', '.join(keyword_names)}")
print("-" * 80)

results = find_shared_sentences(
    results_dir=results_dir,
    keyword_names=keyword_names,
    export=True,
    output_dir=results_dir
)

# Print summary
total_calls = len(results)
calls_with_shared = sum(1 for v in results.values() if v["shared_sentences_count"] > 0)
total_shared = sum(v["shared_sentences_count"] for v in results.values())

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total earnings calls analyzed: {total_calls}")
print(f"Calls with shared sentences: {calls_with_shared}")
print(f"Total shared sentences found: {total_shared}")

if calls_with_shared > 0:
    print("\nCalls with shared sentences:")
    for call_name, data in sorted(results.items()):
        if data["shared_sentences_count"] > 0:
            print(f"  {call_name}: {data['shared_sentences_count']} shared sentences")

