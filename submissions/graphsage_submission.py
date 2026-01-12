"""
ENZYMES-Hard Challenge: GraphSAGE Submission

Trains a lightweight GraphSAGE model (<100K params) on the challenge splits
and writes a predictions CSV for submission.

Usage:
    python submissions/graphsage_submission.py \
        --data_dir data/challenge \
        --output_dir submissions/graphsage \
        --epochs 120
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool
except ImportError as e:
    raise ImportError("PyTorch Geometric is required. Install with: pip install torch-geometric") from e


def parse_args():
    p = argparse.ArgumentParser(description="GraphSAGE submission")
    p.add_argument("--data_dir", type=str, default="data/challenge", help="Path to challenge data")
    p.add_argument("--output_dir", type=str, default="submissions/graphsage", help="Output directory")
    p.add_argument("--epochs", type=int, default=120, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--hidden_dim", type=int, default=64, help="Hidden dim")
    p.add_argument("--num_layers", type=int, default=3, help="GNN layers")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def impute_missing_features(data_list: List[Data]) -> List[Data]:
    """Mean-impute NaNs feature-wise using train/val/test combined."""
    all_feats = []
    for data in data_list:
        x = data.x.numpy()
        mask = ~np.isnan(x)
        all_feats.append(x[mask])
    global_mean = np.concatenate(all_feats).mean() if all_feats else 0.0

    feat_sum, feat_cnt = None, None
    for data in data_list:
        x = data.x.numpy()
        mask = ~np.isnan(x)
        if feat_sum is None:
            feat_sum = np.zeros(x.shape[1])
            feat_cnt = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            feat_sum[j] += x[mask[:, j], j].sum()
            feat_cnt[j] += mask[:, j].sum()
    feat_mean = np.where(feat_cnt > 0, feat_sum / feat_cnt, global_mean)

    imputed = []
    for data in data_list:
        x = data.x.numpy().copy()
        for j in range(x.shape[1]):
            nan_mask = np.isnan(x[:, j])
            x[nan_mask, j] = feat_mean[j]
        new_data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=data.edge_index,
            y=data.y if hasattr(data, "y") else None,
            node_labels=data.node_labels if hasattr(data, "node_labels") else None,
            num_nodes=data.num_nodes,
        )
        if hasattr(data, "graph_id"):
            new_data.graph_id = data.graph_id
        imputed.append(new_data)
    return imputed


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.squeeze() - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1) + 1
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.squeeze().cpu().numpy())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return macro_f1, acc


def predict(model, loader, device) -> List[Tuple[int, int]]:
    model.eval()
    pairs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1) + 1
            for d, p in zip(data.to_data_list(), pred.cpu().numpy()):
                gid = d.graph_id
                if hasattr(gid, "item"):
                    gid = gid.item()
                elif hasattr(gid, "__int__"):
                    gid = int(gid)
                pairs.append((gid, int(p)))
    return pairs


def save_predictions(pairs: List[Tuple[int, int]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_id", "prediction"])
        for gid, pred in sorted(pairs, key=lambda x: x[0]):
            writer.writerow([gid, pred])
    print(f"✅ Saved predictions to {out_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")
    test_data = torch.load(data_dir / "test.pt")

    train_data = impute_missing_features(train_data)
    val_data = impute_missing_features(val_data)
    test_data = impute_missing_features(test_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    in_dim = train_data[0].x.shape[1]
    model = GraphSAGENet(in_dim, args.hidden_dim, args.num_layers, dropout=args.dropout).to(device)
    print(f"Parameters: {model.count_parameters():,} (limit 100,000)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_f1, train_acc = evaluate(model, train_loader, device)
        val_f1, val_acc = evaluate(model, val_loader, device)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | train F1 {train_f1:.3f} | val F1 {val_f1:.3f} | time {elapsed:.1f}s")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    pairs = predict(model, test_loader, device)
    save_predictions(pairs, Path(args.output_dir))


if __name__ == "__main__":
    main()
