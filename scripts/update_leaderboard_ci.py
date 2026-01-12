import argparse
import json
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Path to submission directory")
    parser.add_argument("--team-name", required=True, help="Display name on leaderboard")
    parser.add_argument("--params", default="N/A")
    parser.add_argument("--train-time", default="N/A")
    parser.add_argument("--is-baseline", default="false")
    parser.add_argument("--submitted-at", default="")
    parser.add_argument("--leaderboard", default="web/src/app/leaderboard/data.json")
    args = parser.parse_args()

    submission_dir = Path(args.submission)
    results_path = submission_dir / "evaluation_results.json"
    if not results_path.exists():
        raise SystemExit(f"results file not found: {results_path}")

    data = json.loads(results_path.read_text())
    leaderboard_path = Path(args.leaderboard)
    if not leaderboard_path.exists():
        raise SystemExit(f"Leaderboard data file missing: {leaderboard_path}")

    entries = json.loads(leaderboard_path.read_text())

    submitted_at = args.submitted_at.strip() or datetime.utcnow().isoformat()
    new_entry = {
        "rank": 0,
        "name": args.team_name,
        "macroF1": float(data.get("macro_f1", 0)),
        "accuracy": float(data.get("accuracy", 0)),
        "params": args.params,
        "trainTime": args.train_time,
        "submittedAt": submitted_at,
    }
    if args.is_baseline.lower() == "true":
        new_entry["isBaseline"] = True

    filtered = [e for e in entries if e.get("name") != new_entry["name"]]
    filtered.append(new_entry)
    filtered.sort(key=lambda e: (e.get("macroF1", 0), e.get("accuracy", 0)), reverse=True)
    for idx, entry in enumerate(filtered, start=1):
        entry["rank"] = idx

    leaderboard_path.write_text(json.dumps(filtered, indent=2))

if __name__ == "__main__":
    main()
