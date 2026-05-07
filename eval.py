"""Run a question set against the deployed API and dump answers + latency for manual scoring.

Usage:
    python eval.py --url http://localhost:8000 --testset test_set.json

After it finishes, eval_results.json will contain answers + latencies.
You then manually mark each as correct/incorrect to compute final accuracy.
"""
import argparse
import json
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running API (no trailing slash)")
    parser.add_argument("--testset", default="test_set.json",
                        help="Path to test set JSON")
    parser.add_argument("--out", default="eval_results.json",
                        help="Where to write results")
    args = parser.parse_args()

    with open(args.testset) as f:
        tests = json.load(f)

    results = []
    total_latency = 0.0

    for i, t in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {t['question'][:60]}...")
        start = time.time()
        try:
            r = requests.post(
                f"{args.url}/ask",
                json={"video_id": t["video_id"], "question": t["question"]},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            wall_latency = round(time.time() - start, 2)
            results.append({
                "video_id": t["video_id"],
                "question": t["question"],
                "expected": t.get("expected_answer", ""),
                "actual": data["answer"],
                "server_latency_s": data["latency_seconds"],
                "wall_latency_s": wall_latency,
                "cache_hit": data["cache_hit"],
                "correct": None,  # YOU fill this in manually after reviewing
            })
            total_latency += data["latency_seconds"]
        except Exception as e:
            results.append({
                "video_id": t["video_id"],
                "question": t["question"],
                "error": str(e),
            })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    print(f"\nDone. {n} queries. Avg server latency: {total_latency/n:.2f}s")
    print(f"Results written to {args.out}")
    print("Open it, set 'correct': true/false for each, then count up your accuracy.")


if __name__ == "__main__":
    main()
