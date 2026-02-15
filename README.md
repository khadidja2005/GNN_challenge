# ENZYMES-Hard: Few-Shot Protein Function Classification

<p align="center">
  <img src="https://img.shields.io/badge/Task-Graph%20Classification-blue" alt="Task">
  <img src="https://img.shields.io/badge/Dataset-ENZYMES-green" alt="Dataset">
  <img src="https://img.shields.io/badge/Difficulty-Hard-red" alt="Difficulty">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20Geometric-orange" alt="Framework">
</p>

<p align="center">
  <b>An Educational GNN Challenge for Robust Protein Function Prediction</b>
</p>

---

## Quick Start for Participants

**Want to participate? Follow these steps:**

### Step 1: Set Up Your Environment

```bash
# Clone the repository
git clone https://github.com/khadidja2005/GNN_challenge.git
cd GNN_challenge

# Install dependencies
pip install -r requirements.txt

# Prepare the challenge data
python scripts/prepare_data.py
```

### Step 2: Build Your Model

- Train a GNN model on the 240 training graphs in `data/challenge/train.pt`
- Use the 180 validation graphs in `data/challenge/val.pt` to tune hyperparameters
- Your model must have **≤100K parameters** and train in **≤3 hours on CPU**

```python
# Load the data
import torch
train_graphs = torch.load('data/challenge/train.pt', weights_only=False)
val_graphs = torch.load('data/challenge/val.pt', weights_only=False)
test_graphs = torch.load('data/challenge/test.pt', weights_only=False)
```

### Step 3: Generate Predictions

Run your trained model on the 180 test graphs and save predictions as CSV:

```csv
graph_id,prediction
0,3
1,1
2,5
...
179,2
```

- `graph_id`: 0 to 179 (test graph index)
- `prediction`: 1 to 6 (enzyme class)

Save this file as `predictions.csv`.

### Step 4: Encrypt Your Submission

Your predictions are encrypted before submission so other participants cannot see them.

```bash
# Install encryption library (if not already installed)
pip install cryptography

# Encrypt your predictions
python encryption/encrypt.py predictions.csv encryption/public_key.pem submissions/yourteam.enc
```

Replace `yourteam` with your team name (no spaces, e.g., `alpha_team`).

### Step 5: Submit via Pull Request

1. **Fork** this repository (if you're not a collaborator)
2. **Add** your encrypted file: `submissions/yourteam.enc`
3. **Commit** and push:
   ```bash
   git add submissions/yourteam.enc
   git commit -m "[Submission] YourTeamName"
   git push origin main
   ```
4. **Open a Pull Request** with title: `[Submission] YourTeamName`
5. **Wait** for CI to evaluate (2-5 minutes)

The CI system will:
- Decrypt your submission (only CI has the private key)
- Validate the CSV format
- Evaluate against the hidden test labels
- Comment your score on the PR
- Update the public leaderboard

**Important:** You get only ONE submission. Make sure your model is ready!

---

## What's New
- Encrypted submission system for privacy
- Automated CI evaluation and leaderboard updates
- Ready-to-use GAT/GCN/GraphSAGE submission scripts under `submissions/`
- Web leaderboard at https://khadidja2005.github.io/GNN_challenge/

## Challenge Overview

**ENZYMES-Hard** is a graph classification competition. Your goal is to classify protein structures (represented as graphs) into one of 6 enzyme functional classes.

### Why This Matters
- Enzyme function prediction supports drug discovery
- Better generalization under missing data mirrors real-world lab conditions
- Robust GNNs on protein graphs accelerate annotation of novel enzymes

### The Task

Classify protein graphs into 6 EC top-level enzyme classes:

| Class | Description |
|-------|-------------|
| 1 | Oxidoreductases |
| 2 | Transferases |
| 3 | Hydrolases |
| 4 | Lyases |
| 5 | Isomerases |
| 6 | Ligases |

### What Makes This Hard

| Challenge | Description |
|-----------|-------------|
| Limited Training Data | Only 240 training graphs (40 per class) |
| Imbalanced Validation | Validation set has imbalanced classes |
| Missing Features | 10-15% of node features are NaN |
| Edge Dropout | 10% of edges hidden in test graphs |
| Model Constraints | Max 100K parameters, train in <3h on CPU |

---

## Dataset Statistics

| Split | Graphs | Class Distribution | Notes |
|-------|--------|-------------------|-------|
| Train | 240 | Balanced (40/class) | Complete features |
| Validation | 180 | Imbalanced (45-40-35-25-20-15) | Missing features |
| Test | 180 | Imbalanced (15-20-25-35-40-45) | Missing features + Edge dropout |

### Sample Graph Visualizations (One per Class)

| Class 1 | Class 2 | Class 3 |
|---------|---------|---------|
| ![Class 1](assets/sample_class_1.png) | ![Class 2](assets/sample_class_2.png) | ![Class 3](assets/sample_class_3.png) |

| Class 4 | Class 5 | Class 6 |
|---------|---------|---------|
| ![Class 4](assets/sample_class_4.png) | ![Class 5](assets/sample_class_5.png) | ![Class 6](assets/sample_class_6.png) |

### Graph Properties
- **Nodes per graph**: 2-126 (avg: ~32)
- **Node features**: 18 continuous attributes (chemical/structural properties)
- **Node labels**: 3 categorical labels (amino acid types)
- **Edges**: Represent spatial proximity between amino acids

### Graph Data Specification

Each graph is a PyTorch Geometric `Data` object:

| Component | Attribute | Description |
|-----------|-----------|-------------|
| Adjacency (A) | `data.edge_index` | Edge list in COO format `[2, num_edges]` |
| Node Features (X) | `data.x` | Feature matrix `[num_nodes, 18]` |
| Label (y) | `data.y` | Graph-level class label (1-6) |

```python
import torch
from torch_geometric.data import Data

data_list = torch.load('data/challenge/train.pt', weights_only=False)
graph = data_list[0]

print(f"Adjacency (edge_index): {graph.edge_index.shape}")  # [2, num_edges]
print(f"Node features (x): {graph.x.shape}")                # [num_nodes, 18]
print(f"Label (y): {graph.y.item()}")                       # 1-6
```

---

## Getting Started

### Option 1: Using Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/khadidja2005/GNN_challenge.git
cd GNN_challenge
docker-compose up --build

# Run scripts
docker-compose run gnn python scripts/prepare_data.py
docker-compose run gnn python baselines/simple_gnn.py
```

### Option 2: Local Installation

```bash
git clone https://github.com/khadidja2005/GNN_challenge.git
cd GNN_challenge
pip install -r requirements.txt
python scripts/prepare_data.py
python baselines/simple_gnn.py
```

---

## Repository Structure

```
GNN_challenge/
├── README.md                    # This file
├── RULES.md                     # Detailed challenge rules
├── requirements.txt             # Python dependencies
├── data/challenge/              # Challenge data splits
│   ├── train.pt                 # 240 training graphs
│   ├── val.pt                   # 180 validation graphs
│   └── test.pt                  # 180 test graphs (labels hidden)
├── encryption/                  # Submission encryption
│   ├── encrypt.py               # Encrypt predictions (for participants)
│   ├── decrypt.py               # Decrypt submissions (CI only)
│   └── public_key.pem           # Public key for encryption
├── baselines/
│   └── simple_gnn.py            # Baseline GNN model (<100K params)
├── notebooks/
│   └── getting_started.ipynb    # Starter notebook
├── submissions/                 # Submit your .enc file here
│   ├── template.py              # Submission template
│   └── example_submission.csv   # Example format
└── scripts/
    ├── prepare_data.py          # Data preparation
    ├── evaluate.py              # Evaluation script
    └── validate_submission.py   # Validate CSV format
```

---

## Submission Format

Your `predictions.csv` must have exactly 180 rows:

```csv
graph_id,prediction
0,3
1,1
2,5
...
179,2
```

- `graph_id`: Test graph index (0-179)
- `prediction`: Predicted class (1-6)

### Validate Before Submitting

```bash
python scripts/validate_submission.py --predictions predictions.csv
```

---

## Evaluation

### Primary Metric: Macro F1-Score

$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

This metric treats all classes equally, regardless of frequency.

### Secondary Metric: Accuracy

Used for tiebreaking only.

### Local Evaluation (Validation Set Only)

```bash
python scripts/evaluate.py --predictions predictions.csv --ground_truth val
```

Note: Test labels are hidden. You can only evaluate locally against the validation set.

---

## Rules Summary

1. **Parameter Limit**: Maximum 100,000 trainable parameters
2. **Training Time**: Must complete in <3 hours on CPU
3. **One Submission**: Only ONE submission per participant
4. **No External Data**: Only use the provided training data
5. **No Pre-trained Models**: Train from scratch
6. **Reproducibility**: Set random seed and provide complete code

See [RULES.md](RULES.md) for complete rules.

---

## Tips

**Dealing with Missing Features:**
- Consider imputation (mean, median, learned)
- Use masking to indicate missing values
- Graph-based imputation using neighbor information

**Handling Limited Data:**
- Data augmentation (node dropout, feature noise)
- Regularization (dropout, weight decay)
- Simple architectures often work better

**Class Imbalance:**
- Weighted loss functions
- Focal loss
- Oversampling minority classes

---

## Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [TUDataset Paper](https://arxiv.org/abs/2007.08663)
- [ENZYMES Original Paper](https://academic.oup.com/bioinformatics/article/21/suppl_1/i47/202991)
- [GNN Survey](https://arxiv.org/abs/1901.00596)

---

## Contact

- **Challenge Organizer**: [@khadidja2005](https://github.com/khadidja2005)
- **Issues**: Open an issue for questions or bug reports

---

## License

This challenge uses the ENZYMES dataset from TUDataset. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Good luck!</b>
</p>
