"""
ENZYMES-Hard Challenge: Submission Template

This template shows the structure for a valid submission.
Participants should implement their model and training logic here.

Requirements:
1. Model must have ≤100K parameters
2. Training must complete in ≤5 minutes on CPU
3. Must generate predictions.csv in the correct format

Usage:
    python submissions/template.py
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import PyTorch Geometric
try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    print("Please install PyTorch Geometric: pip install torch-geometric")
    sys.exit(1)


# ============================================================================
# CONFIGURATION - Modify these as needed
# ============================================================================

CONFIG = {
    'seed': 42,
    'data_dir': 'data/challenge',
    'output_dir': 'submissions/my_submission',
    
    # Model hyperparameters
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.5,
    
    # Training hyperparameters
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
}


# ============================================================================
# DATA PREPROCESSING - Implement your feature handling strategy
# ============================================================================

def preprocess_data(data_list):
    """
    Preprocess the data, handling missing features.
    
    This is a simple mean imputation. You can implement more sophisticated methods:
    - Median imputation
    - KNN imputation
    - Learned imputation
    - Graph-based imputation
    - Masking with learnable embeddings
    
    Args:
        data_list: List of PyG Data objects
    
    Returns:
        Preprocessed list of Data objects
    """
    # Compute global feature means for imputation
    all_features = []
    for data in data_list:
        x = data.x.numpy()
        valid_mask = ~np.isnan(x)
        if valid_mask.any():
            all_features.append(x[valid_mask])
    
    if all_features:
        global_mean = np.concatenate(all_features).mean()
    else:
        global_mean = 0.0
    
    # Per-feature mean
    n_features = data_list[0].x.shape[1]
    feature_means = np.full(n_features, global_mean)
    
    for j in range(n_features):
        values = []
        for data in data_list:
            col = data.x[:, j].numpy()
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                values.extend(valid)
        if values:
            feature_means[j] = np.mean(values)
    
    # Apply imputation
    processed = []
    for data in data_list:
        x = data.x.numpy().copy()
        for j in range(n_features):
            nan_mask = np.isnan(x[:, j])
            x[nan_mask, j] = feature_means[j]
        
        new_data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=data.edge_index,
            num_nodes=data.num_nodes
        )
        if hasattr(data, 'y') and data.y is not None:
            new_data.y = data.y
        if hasattr(data, 'graph_id'):
            new_data.graph_id = data.graph_id
        
        processed.append(new_data)
    
    return processed


# ============================================================================
# MODEL DEFINITION - Implement your GNN architecture
# ============================================================================

class MyGNN(nn.Module):
    """
    Your Graph Neural Network implementation.
    
    Requirements:
    - Maximum 100,000 trainable parameters
    - Must handle variable-sized graphs
    - Must output 6-class predictions
    
    Tips for staying under parameter budget:
    - Use smaller hidden dimensions
    - Fewer layers
    - Parameter-efficient architectures (GIN, GraphSAGE)
    - Weight sharing
    """
    
    def __init__(self, in_dim, hidden_dim=64, num_classes=6, num_layers=3, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GNN layers (using GCN as example)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input embedding
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Readout
        x = global_mean_pool(x, batch)
        
        # Classify
        out = self.classifier(x)
        
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINING LOOP - Implement your training strategy
# ============================================================================

def train(model, train_loader, val_loader, config, device):
    """
    Training loop with early stopping.
    
    You can implement:
    - Learning rate scheduling
    - Data augmentation
    - Different loss functions (focal loss, class-weighted loss)
    - Gradient clipping
    - Mixed precision (for GPU)
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            loss = criterion(out, data.y.squeeze() - 1)  # 0-indexed
            
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1) + 1
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.squeeze().cpu().numpy())
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Val F1 = {val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    return model, best_val_f1


# ============================================================================
# PREDICTION - Generate submission file
# ============================================================================

@torch.no_grad()
def generate_predictions(model, test_loader, device):
    """Generate predictions for the test set."""
    model.eval()
    
    predictions = []
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1) + 1  # 1-indexed classes
        
        for i, p in enumerate(pred):
            gid = data.to_data_list()[i].graph_id
            # Handle graph_id whether it's a tensor, numpy array, or int
            if hasattr(gid, 'item'):
                gid = gid.item()
            elif hasattr(gid, '__int__'):
                gid = int(gid)
            predictions.append((gid, p.item()))
    
    return predictions


def save_predictions(predictions, output_path):
    """Save predictions in submission format."""
    with open(output_path, 'w') as f:
        f.write("graph_id,prediction\n")
        for graph_id, pred in sorted(predictions):
            f.write(f"{graph_id},{pred}\n")


# ============================================================================
# MAIN - Run training and generate submission
# ============================================================================

def main():
    print("=" * 60)
    print("🧬 ENZYMES-Hard Challenge - My Submission")
    print("=" * 60)
    
    # Set seed
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    device = torch.device('cpu')  # Required: CPU only
    
    # Load data
    print("\n📂 Loading data...")
    data_dir = Path(CONFIG['data_dir'])
    
    train_data = torch.load(data_dir / 'train.pt')
    val_data = torch.load(data_dir / 'val.pt')
    test_data = torch.load(data_dir / 'test.pt')
    
    # Preprocess
    print("🔧 Preprocessing...")
    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)
    test_data = preprocess_data(test_data)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Create model
    in_dim = train_data[0].x.shape[1]
    model = MyGNN(
        in_dim=in_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=6,
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f"\n🏗️ Model: {num_params:,} parameters")
    
    if num_params > 100000:
        print("❌ ERROR: Model exceeds 100K parameter limit!")
        sys.exit(1)
    
    # Train
    print("\n🏋️ Training...")
    start_time = time.time()
    
    model, best_val_f1 = train(model, train_loader, val_loader, CONFIG, device)
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Training time: {training_time:.1f}s")
    
    if training_time > 300:
        print("❌ WARNING: Training exceeded 5 minute limit!")
    
    print(f"📊 Best validation F1: {best_val_f1:.4f}")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = generate_predictions(model, test_loader, device)
    
    # Save
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'predictions.csv'
    save_predictions(predictions, output_path)
    
    print(f"✅ Predictions saved to: {output_path}")
    print("\n🎉 Done! Submit your predictions.csv file.")


if __name__ == '__main__':
    main()
