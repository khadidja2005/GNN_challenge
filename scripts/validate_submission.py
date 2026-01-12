import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    errors = []
    if 'graph_id' not in df.columns:
        errors.append("Missing 'graph_id' column")
    if 'prediction' not in df.columns:
        errors.append("Missing 'prediction' column")
    if len(df) != 180:
        errors.append(f"Expected 180 predictions, got {len(df)}")
    if 'prediction' in df.columns:
        invalid = df[(df['prediction'] < 1) | (df['prediction'] > 6)]
        if len(invalid) > 0:
            errors.append(f"Invalid predictions (not in 1-6): {invalid['prediction'].unique().tolist()}")
    if 'graph_id' in df.columns and df['graph_id'].duplicated().any():
        errors.append('Duplicate graph_id entries found')

    if errors:
        print('❌ Validation errors:')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('✅ Submission format valid')

if __name__ == "__main__":
    main()
