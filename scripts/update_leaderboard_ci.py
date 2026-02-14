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

    # Check for duplicate submission (one submission per participant)
    existing_names = [e.get("name") for e in entries if not e.get("isBaseline")]
    if args.team_name in existing_names and args.is_baseline.lower() != "true":
        raise SystemExit(f"❌ Submission rejected: '{args.team_name}' has already submitted. Only one submission per participant is allowed.")

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

    # Add new entry (no replacement for non-baselines due to one-submission policy)
    if args.is_baseline.lower() == "true":
        # Baselines can be updated
        entries = [e for e in entries if e.get("name") != new_entry["name"]]
    entries.append(new_entry)

    # Sort by macroF1 (desc), then accuracy (desc) for display order
    entries.sort(key=lambda e: (e.get("macroF1", 0), e.get("accuracy", 0)), reverse=True)

    # Kaggle-style ranking: tied scores share the same rank
    current_rank = 1
    for i, entry in enumerate(entries):
        if i == 0:
            entry["rank"] = current_rank
        else:
            prev = entries[i - 1]
            # Compare macroF1 rounded to avoid float precision issues
            if round(entry.get("macroF1", 0), 6) == round(prev.get("macroF1", 0), 6):
                # Tie: same rank as previous
                entry["rank"] = prev["rank"]
            else:
                # Not a tie: rank = position + 1
                entry["rank"] = i + 1
            current_rank = entry["rank"]

    leaderboard_path.write_text(json.dumps(entries, indent=2))
    print(f"✅ Leaderboard updated: {args.team_name} added at rank {new_entry['rank']}")

if __name__ == "__main__":
    main()
