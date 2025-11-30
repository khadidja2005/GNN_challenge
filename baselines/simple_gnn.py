"""
ENZYMES-Hard Challenge: Simple GNN Baseline

A minimal Graph Neural Network baseline that:
- Uses <100K parameters
- Trains in <5 minutes on CPU
- Handles missing features with simple imputation
- Achieves reasonable performance for comparison

This serves as a reference implementation for challenge participants.

Usage:
    python baselines/simple_gnn.py [--epochs 100] [--hidden_dim 64]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
from sklearn.metrics import f1_score, accuracy_score

# Try to import PyTorch Geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ PyTorch Geometric not found. Please install it:")
    print("   pip install torch-geometric")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train simple GNN baseline')
    parser.add_argument('--data_dir', type=str, default='data/challenge',
                        help='Path to challenge data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='submissions/baseline',
                        help='Output directory for predictions')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def impute_missing_features(data_list: list) -> list:
    """
    Simple mean imputation for missing (NaN) features.
    
    For the challenge, participants can implement more sophisticated methods:
    - Median imputation
    - KNN imputation
    - Learned imputation
    - Graph-based imputation using neighbor features
    """
    # Collect all features to compute global mean
    all_features = []
    for data in data_list:
        x = data.x.numpy()
        mask = ~np.isnan(x)
        all_features.append(x[mask])
    
    if all_features:
        global_mean = np.concatenate(all_features).mean()
    else:
        global_mean = 0.0
    
    # Compute per-feature mean
    feature_sums = None
    feature_counts = None
    
    for data in data_list:
        x = data.x.numpy()
        mask = ~np.isnan(x)
        
        if feature_sums is None:
            feature_sums = np.zeros(x.shape[1])
            feature_counts = np.zeros(x.shape[1])
        
        for j in range(x.shape[1]):
            col_mask = mask[:, j]
            feature_sums[j] += x[col_mask, j].sum()
            feature_counts[j] += col_mask.sum()
    
    # Compute feature means (use global mean if no valid values)
    feature_means = np.where(
        feature_counts > 0,
        feature_sums / feature_counts,
        global_mean
    )
    
    # Impute missing values
    imputed_data = []
    for data in data_list:
        x = data.x.numpy().copy()
        
        for j in range(x.shape[1]):
            nan_mask = np.isnan(x[:, j])
            x[nan_mask, j] = feature_means[j]
        
        # Create new data object with imputed features
        new_data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=data.edge_index,
            y=data.y if hasattr(data, 'y') else None,
            node_labels=data.node_labels if hasattr(data, 'node_labels') else None,
            num_nodes=data.num_nodes
        )
        if hasattr(data, 'graph_id'):
            new_data.graph_id = data.graph_id
        
        imputed_data.append(new_data)
    
    return imputed_data


class MLP(nn.Module):
    """Simple MLP for GIN."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class SimpleGNN(nn.Module):
    """
    Simple Graph Isomorphism Network (GIN) for graph classification.
    
    Architecture:
    - Node feature embedding
    - Multiple GIN layers
    - Global pooling
    - Classification head
    
    This model is designed to stay under 100K parameters.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 6,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input embedding
        self.input_embed = nn.Linear(in_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input embedding
        x = self.input_embed(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        # Adjust labels to 0-indexed
        loss = criterion(out, data.y.squeeze() - 1)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, list, list]:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1) + 1  # Convert back to 1-indexed
        
        all_preds.extend(pred.cpu().numpy())
        if hasattr(data, 'y') and data.y is not None:
            all_labels.extend(data.y.squeeze().cpu().numpy())
    
    if all_labels:
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        macro_f1, accuracy = 0.0, 0.0
    
    return macro_f1, accuracy, all_preds, all_labels


@torch.no_grad()
def predict(model, loader, device) -> list:
    """Generate predictions for test set."""
    model.eval()
    
    all_preds = []
    all_graph_ids = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1) + 1  # Convert back to 1-indexed
        
        all_preds.extend(pred.cpu().numpy())
        for d in data.to_data_list():
            # Handle graph_id whether it's a tensor, numpy array, or int
            gid = d.graph_id
            if hasattr(gid, 'item'):
                gid = gid.item()
            elif hasattr(gid, '__int__'):
                gid = int(gid)
            all_graph_ids.append(gid)
    
    return list(zip(all_graph_ids, all_preds))


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🧬 ENZYMES-Hard: Simple GNN Baseline")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cpu')  # Challenge requires CPU training
    print(f"\n🖥️ Using device: {device}")
    
    # Load data
    print("\n📂 Loading challenge data...")
    data_dir = Path(args.data_dir)
    
    train_data = torch.load(data_dir / 'train.pt')
    val_data = torch.load(data_dir / 'val.pt')
    test_data = torch.load(data_dir / 'test.pt')
    
    print(f"   Train: {len(train_data)} graphs")
    print(f"   Val:   {len(val_data)} graphs")
    print(f"   Test:  {len(test_data)} graphs")
    
    # Impute missing features
    print("\n🔧 Imputing missing features...")
    train_data = impute_missing_features(train_data)
    val_data = impute_missing_features(val_data)
    test_data = impute_missing_features(test_data)
    print("   ✓ Missing features imputed with mean values")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Get input dimension
    in_dim = train_data[0].x.shape[1]
    print(f"\n📊 Node feature dimension: {in_dim}")
    
    # Create model
    model = SimpleGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=6,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"\n🏗️ Model created:")
    print(f"   Architecture: GIN ({args.num_layers} layers, {args.hidden_dim} hidden)")
    print(f"   Parameters: {num_params:,} (limit: 100,000)")
    
    if num_params > 100000:
        print("   ⚠️ WARNING: Model exceeds parameter limit!")
    else:
        print("   ✓ Within parameter limit")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"\n🏋️ Training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_val_f1 = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        train_f1, train_acc, _, _ = evaluate(model, train_loader, device)
        val_f1, val_acc, _, _ = evaluate(model, val_loader, device)
        
        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    training_time = time.time() - start_time
    print("-" * 60)
    print(f"\n⏱️ Total training time: {training_time:.1f}s")
    
    if training_time > 300:
        print("   ⚠️ WARNING: Training exceeded 5 minute limit!")
    else:
        print("   ✓ Within time limit")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Final evaluation
    print(f"\n📊 Best model (epoch {best_epoch}):")
    val_f1, val_acc, _, _ = evaluate(model, val_loader, device)
    print(f"   Validation Macro F1: {val_f1:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")
    
    # Generate test predictions
    print("\n🔮 Generating test predictions...")
    predictions = predict(model, test_loader, device)
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = output_dir / 'predictions.csv'
    with open(predictions_file, 'w') as f:
        f.write("graph_id,prediction\n")
        for graph_id, pred in sorted(predictions):
            f.write(f"{graph_id},{pred}\n")
    
    print(f"   ✓ Predictions saved to: {predictions_file}")
    
    # Save model info
    info = {
        'model': 'SimpleGNN (GIN)',
        'parameters': num_params,
        'training_time_seconds': training_time,
        'best_epoch': best_epoch,
        'val_macro_f1': float(val_f1),
        'val_accuracy': float(val_acc),
        'hyperparameters': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seed': args.seed
        }
    }
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✓ Model info saved to: {output_dir / 'model_info.json'}")
    
    print("\n" + "=" * 60)
    print("✅ Baseline training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Check predictions: {predictions_file}")
    print(f"  2. Evaluate on validation: python scripts/evaluate.py --predictions {predictions_file} --ground_truth val")
    print(f"  3. Improve the model and beat the baseline!")
    
    # Cleanup
    os.remove('best_model.pt')


if __name__ == '__main__':
    main()
