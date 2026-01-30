"""
ENZYMES-Hard Challenge: Evaluation Script

This script evaluates predictions against the hidden test set labels.
Participants can use this to validate their submission format on the validation set.

Usage:
    python scripts/evaluate.py --predictions submissions/your_predictions.csv
    python scripts/evaluate.py --predictions submissions/your_predictions.csv --ground_truth val
"""

import os
import sys
import json
import base64
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate challenge predictions')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--ground_truth', type=str, default='test',
                        choices=['test', 'val'],
                        help='Which ground truth to evaluate against')
    parser.add_argument('--data_dir', type=str, default='data/challenge',
                        help='Path to challenge data directory')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed per-class metrics')
    return parser.parse_args()


def load_predictions(filepath: str) -> Dict[int, int]:
    """
    Load predictions from CSV file.
    
    Expected format:
        graph_id,prediction
        0,3
        1,1
        ...
    """
    df = pd.read_csv(filepath)
    
    # Validate columns
    required_cols = {'graph_id', 'prediction'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_cols}. Found: {set(df.columns)}")
    
    # Validate prediction values
    valid_classes = {1, 2, 3, 4, 5, 6}
    invalid_preds = set(df['prediction'].unique()) - valid_classes
    if invalid_preds:
        raise ValueError(f"Invalid prediction values: {invalid_preds}. Must be in {valid_classes}")
    
    predictions = dict(zip(df['graph_id'].astype(int), df['prediction'].astype(int)))
    
    return predictions


def load_ground_truth(data_dir: str, split: str) -> Dict[int, int]:
    """
    Load ground truth labels.
    """
    data_dir = Path(data_dir)
    
    if split == 'test':
        # Prefer labels supplied via secret (Base64 or raw JSON)
        secret_b64 = os.environ.get('TEST_LABELS_B64', '').strip()
        secret_json = os.environ.get('TEST_LABELS_JSON', '').strip()
        labels = None

        if secret_b64:
            try:
                decoded = base64.b64decode(secret_b64)
                labels = json.loads(decoded)
            except Exception as e:
                raise ValueError(f"Failed to decode TEST_LABELS_B64: {e}") from e
        elif secret_json:
            try:
                labels = json.loads(secret_json)
            except Exception as e:
                raise ValueError(f"Failed to parse TEST_LABELS_JSON: {e}") from e
        else:
            # Fallback: file only present in CI/private runs
            gt_file = data_dir / '.ground_truth' / 'test_labels.json'
            if not gt_file.exists():
                raise FileNotFoundError(
                    f"Ground truth file not found: {gt_file}\n"
                    "Test labels are hidden; use the validation set locally."
                )
            with open(gt_file, 'r') as f:
                labels = json.load(f)
        # Convert string keys to int
        return {int(k): int(v) for k, v in labels.items()}
    
    elif split == 'val':
        # Load from validation data (allowlist PyG Data in newer torch)
        import torch
        try:
            from torch_geometric.data import Data as _PyGData
            from torch_geometric.data import Batch as _PyGBatch
            torch.serialization.add_safe_globals([_PyGData, _PyGBatch])  # safe for our local artifact
        except Exception:
            pass
        val_data = torch.load(data_dir / 'val.pt', weights_only=False)
        return {i: int(data.y.item()) for i, data in enumerate(val_data)}
    
    else:
        raise ValueError(f"Unknown split: {split}")


def evaluate(predictions: Dict[int, int], ground_truth: Dict[int, int]) -> Dict:
    """
    Compute evaluation metrics.
    """
    # Align predictions with ground truth
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    
    if len(common_ids) != len(ground_truth):
        missing = set(ground_truth.keys()) - set(predictions.keys())
        raise ValueError(f"Missing predictions for graph IDs: {missing}")
    
    y_true = [ground_truth[i] for i in sorted(common_ids)]
    y_pred = [predictions[i] for i in sorted(common_ids)]
    
    # Compute metrics
    results = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro'),
        'num_predictions': len(common_ids),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6]).tolist(),
        'per_class': {}
    }
    
    # Per-class metrics
    for cls in range(1, 7):
        cls_true = [1 if y == cls else 0 for y in y_true]
        cls_pred = [1 if y == cls else 0 for y in y_pred]
        
        results['per_class'][cls] = {
            'f1': f1_score(cls_true, cls_pred),
            'precision': precision_score(cls_true, cls_pred, zero_division=0),
            'recall': recall_score(cls_true, cls_pred, zero_division=0),
            'support': sum(cls_true)
        }
    
    return results


def print_results(results: Dict, detailed: bool = False):
    """
    Print evaluation results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🏆 Primary Metric (Macro F1):  {results['macro_f1']:.4f}")
    print(f"📈 Secondary Metric (Accuracy): {results['accuracy']:.4f}")
    
    print(f"\n📋 Additional Metrics:")
    print(f"   Weighted F1:     {results['weighted_f1']:.4f}")
    print(f"   Macro Precision: {results['macro_precision']:.4f}")
    print(f"   Macro Recall:    {results['macro_recall']:.4f}")
    print(f"   Predictions:     {results['num_predictions']}")
    
    if detailed:
        print("\n📊 Per-Class Metrics:")
        print("-" * 50)
        print(f"{'Class':<8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
        print("-" * 50)
        
        class_names = {
            1: 'Oxidored', 2: 'Transfer', 3: 'Hydrolas',
            4: 'Lyases', 5: 'Isomeras', 6: 'Ligases'
        }
        
        for cls in range(1, 7):
            m = results['per_class'][cls]
            print(f"{cls} ({class_names[cls][:7]})"[:8].ljust(8) + 
                  f"{m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['support']:>8}")
        
        print("\n📉 Confusion Matrix:")
        print("   (rows=true, cols=predicted)")
        cm = np.array(results['confusion_matrix'])
        print("      " + "  ".join([f"{i:>3}" for i in range(1, 7)]))
        for i, row in enumerate(cm):
            print(f"   {i+1}: " + "  ".join([f"{v:>3}" for v in row]))
    
    print("\n" + "=" * 60)


def validate_submission_format(predictions_file: str, expected_count: int):
    """
    Validate that submission file has correct format.
    """
    df = pd.read_csv(predictions_file)
    
    errors = []
    
    # Check columns
    if 'graph_id' not in df.columns:
        errors.append("Missing 'graph_id' column")
    if 'prediction' not in df.columns:
        errors.append("Missing 'prediction' column")
    
    # Check row count
    if len(df) != expected_count:
        errors.append(f"Expected {expected_count} predictions, got {len(df)}")
    
    # Check for duplicates
    if df['graph_id'].duplicated().any():
        errors.append("Duplicate graph_id entries found")
    
    # Check prediction range
    if 'prediction' in df.columns:
        invalid = df[(df['prediction'] < 1) | (df['prediction'] > 6)]
        if len(invalid) > 0:
            errors.append(f"Invalid predictions (not in 1-6): {invalid['prediction'].unique().tolist()}")
    
    # Check graph_id range
    if 'graph_id' in df.columns:
        expected_ids = set(range(expected_count))
        actual_ids = set(df['graph_id'])
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            errors.append(f"Missing graph_ids: {sorted(missing)[:10]}...")
        if extra:
            errors.append(f"Extra graph_ids: {sorted(extra)[:10]}...")
    
    return errors


def main():
    args = parse_args()
    
    print("\n🔍 Loading predictions...")
    
    # Validate file exists
    if not os.path.exists(args.predictions):
        print(f"❌ Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    # Load and validate predictions
    try:
        predictions = load_predictions(args.predictions)
        print(f"   ✓ Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        sys.exit(1)
    
    # Validate format
    expected_count = 180 if args.ground_truth == 'test' else 180
    errors = validate_submission_format(args.predictions, expected_count)
    if errors:
        print("\n⚠️ Submission format errors:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)
    print("   ✓ Submission format valid")
    
    # Load ground truth
    print(f"\n🔍 Loading ground truth ({args.ground_truth} set)...")
    try:
        ground_truth = load_ground_truth(args.data_dir, args.ground_truth)
        print(f"   ✓ Loaded {len(ground_truth)} ground truth labels")
    except FileNotFoundError as e:
        if args.ground_truth == 'test':
            print("❌ Test set ground truth is not available to participants.")
            print("   Use --ground_truth val to evaluate on validation set.")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Evaluate
    print("\n📊 Computing metrics...")
    try:
        results = evaluate(predictions, ground_truth)
    except ValueError as e:
        print(f"❌ Evaluation error: {e}")
        sys.exit(1)
    
    # Print results
    print_results(results, detailed=args.detailed)
    
    # Save results
    results_file = Path(args.predictions).parent / 'evaluation_results.json'
    # Convert numpy types for JSON
    json_results = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) 
        for k, v in results.items()
    }
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == '__main__':
    main()
