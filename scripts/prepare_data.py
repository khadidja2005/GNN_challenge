"""
ENZYMES-Hard Challenge: Data Preparation Script

This script processes the original ENZYMES dataset from TUDataset and creates
the challenge splits with difficulty modifications:
1. Limited training data (240 graphs, 40 per class)
2. Imbalanced validation set (60-40-30-20-20-10 distribution)
3. Missing features (10-15% NaN values)
4. Edge dropout (10% edges hidden in test set)

Usage:
    python scripts/prepare_data.py [--seed 42] [--output_dir data/challenge]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ENZYMES-Hard challenge data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='Dataset/ENZYMES', 
                        help='Path to original ENZYMES dataset')
    parser.add_argument('--output_dir', type=str, default='data/challenge',
                        help='Output directory for challenge data')
    parser.add_argument('--missing_rate', type=float, default=0.12,
                        help='Rate of missing features (default: 0.12 = 12%)')
    parser.add_argument('--edge_dropout', type=float, default=0.10,
                        help='Rate of edge dropout for test set (default: 0.10 = 10%)')
    return parser.parse_args()


def load_tudataset(data_dir: str) -> Dict:
    """
    Load ENZYMES dataset from TUDataset format.
    
    Returns:
        Dictionary containing all dataset components
    """
    data_dir = Path(data_dir)
    
    print("📂 Loading ENZYMES dataset...")
    
    # Load adjacency list (edges)
    edges = []
    with open(data_dir / 'ENZYMES_A.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            src, dst = int(parts[0].strip()), int(parts[1].strip())
            edges.append((src, dst))
    edges = np.array(edges)
    print(f"   ✓ Loaded {len(edges)} edges")
    
    # Load graph indicators (which graph each node belongs to)
    graph_indicator = []
    with open(data_dir / 'ENZYMES_graph_indicator.txt', 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()))
    graph_indicator = np.array(graph_indicator)
    print(f"   ✓ Loaded {len(graph_indicator)} nodes across {graph_indicator.max()} graphs")
    
    # Load graph labels
    graph_labels = []
    with open(data_dir / 'ENZYMES_graph_labels.txt', 'r') as f:
        for line in f:
            graph_labels.append(int(line.strip()))
    graph_labels = np.array(graph_labels)
    print(f"   ✓ Loaded {len(graph_labels)} graph labels")
    
    # Load node labels
    node_labels = []
    with open(data_dir / 'ENZYMES_node_labels.txt', 'r') as f:
        for line in f:
            node_labels.append(int(line.strip()))
    node_labels = np.array(node_labels)
    print(f"   ✓ Loaded {len(node_labels)} node labels")
    
    # Load node attributes
    node_attributes = []
    with open(data_dir / 'ENZYMES_node_attributes.txt', 'r') as f:
        for line in f:
            attrs = [float(x.strip()) for x in line.strip().split(',')]
            node_attributes.append(attrs)
    node_attributes = np.array(node_attributes)
    print(f"   ✓ Loaded node attributes with shape {node_attributes.shape}")
    
    return {
        'edges': edges,
        'graph_indicator': graph_indicator,
        'graph_labels': graph_labels,
        'node_labels': node_labels,
        'node_attributes': node_attributes
    }


def build_graph_list(data: Dict) -> List[Dict]:
    """
    Convert TUDataset format to a list of individual graph dictionaries.
    """
    print("\n🔧 Building graph list...")
    
    num_graphs = data['graph_labels'].shape[0]
    graphs = []
    
    for graph_id in range(1, num_graphs + 1):
        # Get nodes belonging to this graph
        node_mask = data['graph_indicator'] == graph_id
        node_indices = np.where(node_mask)[0]
        
        # Create node index mapping (global to local)
        global_to_local = {global_idx + 1: local_idx 
                          for local_idx, global_idx in enumerate(node_indices)}
        
        # Get edges for this graph
        edge_mask = np.isin(data['edges'][:, 0], node_indices + 1) & \
                    np.isin(data['edges'][:, 1], node_indices + 1)
        graph_edges = data['edges'][edge_mask]
        
        # Convert to local indices
        local_edges = np.array([[global_to_local[e[0]], global_to_local[e[1]]] 
                                for e in graph_edges])
        
        # Get node features
        node_features = data['node_attributes'][node_mask]
        node_labels = data['node_labels'][node_mask]
        
        graphs.append({
            'graph_id': graph_id - 1,  # 0-indexed
            'label': data['graph_labels'][graph_id - 1],
            'edge_index': local_edges.T if len(local_edges) > 0 else np.zeros((2, 0), dtype=int),
            'node_features': node_features,
            'node_labels': node_labels,
            'num_nodes': len(node_indices),
            'num_edges': len(graph_edges)
        })
    
    print(f"   ✓ Built {len(graphs)} graphs")
    
    return graphs


def create_splits(graphs: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits with specified distributions.
    
    Train: 240 graphs (40 per class, balanced)
    Val: 180 graphs (imbalanced: 60-40-30-20-20-10)
    Test: 180 graphs (balanced: 30 per class)
    """
    print("\n✂️ Creating data splits...")
    np.random.seed(seed)
    
    # Group graphs by class
    class_graphs = defaultdict(list)
    for graph in graphs:
        class_graphs[graph['label']].append(graph)
    
    # Shuffle within each class
    for label in class_graphs:
        np.random.shuffle(class_graphs[label])
    
    # Print class distribution
    print("   Original class distribution:")
    for label in sorted(class_graphs.keys()):
        print(f"      Class {label}: {len(class_graphs[label])} graphs")
    
    train_graphs = []
    val_graphs = []
    test_graphs = []
    
    # Imbalanced validation distribution (total: 180 graphs)
    # Distribution that sums to 180 and leaves 30 per class for test
    # With 60 remaining per class: val + test <= 60
    # val: 50+40+30+25+20+15 = 180, test: 10+20+30+35+40+45 = 180... doesn't work
    # Better: val: 45+38+32+28+22+15 = 180, but we need 30 test each
    # Actually: each class has 60, we need 30 for test, so max 30 for val per class
    # Imbalanced val (capped at 30): 30+30+30+30+30+30 = 180 (must be balanced)
    # OR reduce test to allow imbalance in val
    # Let's do: train=40, val=variable (imbalanced), test=variable (what's left)
    
    # New approach: 
    # Train: 40 per class = 240 (balanced)
    # Val: 45+40+35+25+20+15 = 180 (imbalanced)  
    # Test: 15+20+25+35+40+45 = 180 (what remains)
    val_target = {1: 45, 2: 40, 3: 35, 4: 25, 5: 20, 6: 15}  # = 180 total
    
    # First pass: assign train and collect remaining for val/test pool
    remaining_pool = defaultdict(list)
    for label in sorted(class_graphs.keys()):
        available = class_graphs[label]
        
        # Train: 40 per class (balanced)
        train_graphs.extend(available[:40])
        
        # Remaining 60 graphs go to val/test pool
        remaining_pool[label] = available[40:]
    
    # Assign validation (imbalanced) and test (remaining)
    for label in sorted(remaining_pool.keys()):
        pool = remaining_pool[label]
        np.random.shuffle(pool)
        
        val_count = val_target[label]
        
        val_graphs.extend(pool[:val_count])
        test_graphs.extend(pool[val_count:])
    
    # Shuffle splits
    np.random.shuffle(train_graphs)
    np.random.shuffle(val_graphs)
    np.random.shuffle(test_graphs)
    
    # Update graph IDs within splits
    for i, g in enumerate(train_graphs):
        g['split_id'] = i
    for i, g in enumerate(val_graphs):
        g['split_id'] = i
    for i, g in enumerate(test_graphs):
        g['split_id'] = i
    
    print(f"\n   Split sizes:")
    print(f"      Train: {len(train_graphs)} graphs")
    print(f"      Val:   {len(val_graphs)} graphs")
    print(f"      Test:  {len(test_graphs)} graphs")
    
    # Print split class distributions
    for split_name, split_graphs in [('Train', train_graphs), ('Val', val_graphs), ('Test', test_graphs)]:
        dist = defaultdict(int)
        for g in split_graphs:
            dist[g['label']] += 1
        dist_str = ', '.join([f"{k}:{v}" for k, v in sorted(dist.items())])
        print(f"      {split_name} distribution: {dist_str}")
    
    return train_graphs, val_graphs, test_graphs


def add_missing_features(graphs: List[Dict], missing_rate: float, seed: int) -> List[Dict]:
    """
    Add missing features (NaN values) to node features.
    """
    print(f"\n❓ Adding missing features (rate: {missing_rate:.1%})...")
    np.random.seed(seed)
    
    total_features = 0
    total_missing = 0
    
    for graph in graphs:
        features = graph['node_features'].copy().astype(float)
        num_elements = features.size
        num_missing = int(num_elements * missing_rate)
        
        # Randomly select positions to make NaN
        flat_indices = np.random.choice(num_elements, num_missing, replace=False)
        row_indices = flat_indices // features.shape[1]
        col_indices = flat_indices % features.shape[1]
        features[row_indices, col_indices] = np.nan
        
        graph['node_features'] = features
        total_features += num_elements
        total_missing += num_missing
    
    print(f"   ✓ Added {total_missing:,} missing values ({total_missing/total_features:.1%} of features)")
    
    return graphs


def apply_edge_dropout(graphs: List[Dict], dropout_rate: float, seed: int) -> List[Dict]:
    """
    Remove a percentage of edges from graphs.
    """
    print(f"\n🔗 Applying edge dropout (rate: {dropout_rate:.1%})...")
    np.random.seed(seed)
    
    total_edges = 0
    dropped_edges = 0
    
    for graph in graphs:
        edge_index = graph['edge_index']
        num_edges = edge_index.shape[1]
        
        if num_edges > 0:
            # Keep edges with probability (1 - dropout_rate)
            keep_mask = np.random.random(num_edges) > dropout_rate
            graph['edge_index'] = edge_index[:, keep_mask]
            graph['original_num_edges'] = num_edges
            
            total_edges += num_edges
            dropped_edges += num_edges - keep_mask.sum()
    
    print(f"   ✓ Dropped {dropped_edges:,} edges ({dropped_edges/total_edges:.1%} of edges)")
    
    return graphs


def convert_to_pytorch(graphs: List[Dict], include_labels: bool = True) -> List:
    """
    Convert graph dictionaries to PyTorch tensors.
    """
    try:
        from torch_geometric.data import Data
        use_pyg = True
    except ImportError:
        use_pyg = False
    
    pytorch_graphs = []
    
    for graph in graphs:
        # Convert features (handle NaN by keeping as float tensor)
        x = torch.tensor(graph['node_features'], dtype=torch.float32)
        
        # Convert edge index
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        
        # Convert node labels (as additional features)
        node_labels = torch.tensor(graph['node_labels'], dtype=torch.long)
        
        if use_pyg:
            data = Data(
                x=x,
                edge_index=edge_index,
                node_labels=node_labels,
                num_nodes=graph['num_nodes']
            )
            if include_labels:
                data.y = torch.tensor([graph['label']], dtype=torch.long)
            data.graph_id = graph['split_id']
        else:
            data = {
                'x': x,
                'edge_index': edge_index,
                'node_labels': node_labels,
                'num_nodes': graph['num_nodes'],
                'graph_id': graph['split_id']
            }
            if include_labels:
                data['y'] = torch.tensor([graph['label']], dtype=torch.long)
        
        pytorch_graphs.append(data)
    
    return pytorch_graphs


def save_split(graphs: List, filepath: str):
    """
    Save a split to disk.
    """
    torch.save(graphs, filepath)
    print(f"   ✓ Saved {len(graphs)} graphs to {filepath}")


def create_metadata(train_graphs, val_graphs, test_graphs, args) -> Dict:
    """
    Create metadata JSON for the challenge.
    """
    metadata = {
        'challenge_name': 'ENZYMES-Hard: Few-Shot Protein Function Classification',
        'version': '1.0',
        'seed': args.seed,
        'num_classes': 6,
        'class_names': {
            '1': 'Oxidoreductases',
            '2': 'Transferases',
            '3': 'Hydrolases',
            '4': 'Lyases',
            '5': 'Isomerases',
            '6': 'Ligases'
        },
        'splits': {
            'train': {
                'num_graphs': len(train_graphs),
                'distribution': 'balanced (40 per class)',
                'missing_features': False,
                'edge_dropout': False
            },
            'val': {
                'num_graphs': len(val_graphs),
                'distribution': 'imbalanced (45-40-35-25-20-15)',
                'missing_features': True,
                'missing_rate': args.missing_rate,
                'edge_dropout': False
            },
            'test': {
                'num_graphs': len(test_graphs),
                'distribution': 'imbalanced (15-20-25-35-40-45)',
                'missing_features': True,
                'missing_rate': args.missing_rate,
                'edge_dropout': True,
                'edge_dropout_rate': args.edge_dropout
            }
        },
        'node_features': {
            'dimension': 18,
            'description': 'Chemical and structural properties of amino acids'
        },
        'node_labels': {
            'num_classes': 3,
            'description': 'Amino acid type categories'
        },
        'constraints': {
            'max_parameters': 100000,
            'max_training_time_seconds': 300,
            'device': 'cpu'
        },
        'evaluation': {
            'primary_metric': 'macro_f1',
            'secondary_metric': 'accuracy'
        }
    }
    
    return metadata


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🧬 ENZYMES-Hard Challenge Data Preparation")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original dataset
    data = load_tudataset(args.data_dir)
    
    # Build graph list
    graphs = build_graph_list(data)
    
    # Create splits
    train_graphs, val_graphs, test_graphs = create_splits(graphs, args.seed)
    
    # Add missing features to val and test
    val_graphs = add_missing_features(val_graphs, args.missing_rate, args.seed + 1)
    test_graphs = add_missing_features(test_graphs, args.missing_rate, args.seed + 2)
    
    # Apply edge dropout to test only
    test_graphs = apply_edge_dropout(test_graphs, args.edge_dropout, args.seed + 3)
    
    # Store ground truth labels for test set (for organizers only)
    # Convert to Python int to ensure JSON serialization works
    test_labels = {int(g['split_id']): int(g['label']) for g in test_graphs}
    
    # Convert to PyTorch format
    print("\n📦 Converting to PyTorch format...")
    train_pytorch = convert_to_pytorch(train_graphs, include_labels=True)
    val_pytorch = convert_to_pytorch(val_graphs, include_labels=True)
    test_pytorch = convert_to_pytorch(test_graphs, include_labels=False)  # Hide test labels
    
    # Save splits
    print("\n💾 Saving challenge data...")
    save_split(train_pytorch, output_dir / 'train.pt')
    save_split(val_pytorch, output_dir / 'val.pt')
    save_split(test_pytorch, output_dir / 'test.pt')
    
    # Save ground truth (hidden from participants)
    gt_dir = output_dir / '.ground_truth'
    gt_dir.mkdir(exist_ok=True)
    with open(gt_dir / 'test_labels.json', 'w') as f:
        json.dump(test_labels, f, indent=2)
    print(f"   ✓ Saved ground truth to {gt_dir / 'test_labels.json'} (keep hidden!)")
    
    # Create and save metadata
    metadata = create_metadata(train_graphs, val_graphs, test_graphs, args)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata to {output_dir / 'metadata.json'}")
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print(f"\nChallenge data saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Run the baseline: python baselines/simple_gnn.py")
    print("  2. Explore the data: jupyter notebook notebooks/getting_started.ipynb")
    print("  3. Build your model and submit predictions!")


if __name__ == '__main__':
    main()
