import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

# ==========================================
# 1. HELICAL PHYSICS GRAPH DATASET
# ==========================================
NUCLEOTIDE_MAP = {'A': [1.0, 0, 0, 0], 'C': [0, 1.0, 0, 0], 'G': [0, 0, 1.0, 0], 'T': [0, 0, 0, 1.0], 'N': [0.25, 0.25, 0.25, 0.25]}

class RealLoopSeqDataset(Dataset):
    def __init__(self, txt_file):
        super(RealLoopSeqDataset, self).__init__()
        self.df = pd.read_csv(txt_file, sep='\t')
        self.df.columns = self.df.columns.str.strip()
        self.seq_col = 'Sequence'
        self.target_col = 'C0'
        self.df = self.df.dropna(subset=[self.seq_col, self.target_col]).reset_index(drop=True)
        print(f"Loaded {len(self.df)} sequences from {txt_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx][self.seq_col].upper()
        c_score = float(self.df.iloc[idx][self.target_col])
        
        x = [NUCLEOTIDE_MAP.get(base, NUCLEOTIDE_MAP['N']) for base in seq]
        x = torch.tensor(x, dtype=torch.float)
        
        seq_len = len(seq)
        sources, targets, edge_attrs = [], [], []
        for i in range(seq_len):
            if i < seq_len - 1:   
                sources.extend([i, i+1])
                targets.extend([i+1, i])
                edge_attrs.extend([[1.0], [1.0]])
            if i < seq_len - 10:  
                sources.extend([i, i+10])
                targets.extend([i+10, i])
                edge_attrs.extend([[0.2], [0.2]])
                
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        y = torch.tensor([c_score], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
# ==========================================
# 2. THE CONTINUOUS CIRCLE-DOT FORMER ARCHITECTURE
# ==========================================
class HelicalGNNFrontend(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.embed(x)
        x = self.act(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.act(self.conv2(x, edge_index, edge_attr=edge_attr))
        return x

class HelicalDynamicsFunc(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.Tanh(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    def forward(self, t, y):
        return self.net(y)

class ContinuousODEBlock(nn.Module):
    def __init__(self, odefunc, integration_time=1.0):
        super().__init__()
        self.odefunc = odefunc
        # OPTIMIZATION: register as buffer so it moves to GPU with .to(device) automatically
        self.register_buffer('integration_time', torch.tensor([0.0, integration_time]).float())

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_time, method='rk4', atol=1e-4, rtol=1e-4)
        return out[1]

class CircleDotFormer(nn.Module):
    def __init__(self, node_dim=4, edge_dim=1, hidden_dim=64):
        super().__init__()
        self.gnn = HelicalGNNFrontend(node_dim, edge_dim, hidden_dim)
        self.ode = ContinuousODEBlock(HelicalDynamicsFunc(hidden_dim))
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index, data.edge_attr)
        x_evolved = self.ode(x)
        graph_mean = global_mean_pool(x_evolved, data.batch)
        graph_max = global_max_pool(x_evolved, data.batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        return self.predictor(graph_repr).squeeze(-1)

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_helical_model(data_path, epochs=10, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # OPTIMIZATION: enable TF32 on Ampere+ GPUs for faster matmuls with minimal precision loss
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # auto-tune kernels for fixed input shapes

    print("Preparing Helical Graphs...")
    dataset = RealLoopSeqDataset(data_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # OPTIMIZATION: pin_memory + num_workers for async CPU→GPU transfer
    # persistent_workers avoids respawning workers every epoch
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    model = CircleDotFormer(node_dim=4, edge_dim=1, hidden_dim=64).to(device)

    # OPTIMIZATION: torch.compile() fuses ops (Linux/Mac only — requires Triton, not available on Windows)
    import platform
    if platform.system() != 'Windows':
        try:
            model = torch.compile(model)
            print("torch.compile() enabled — using fused kernels")
        except Exception as e:
            print(f"torch.compile() not available ({e}), continuing without it")
    else:
        print("torch.compile() skipped (not supported on Windows — requires Triton)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # OPTIMIZATION: AMP (Automatic Mixed Precision) — uses FP16 on GPU Tensor Cores
    # GradScaler prevents underflow in FP16 gradients
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    autocast_ctx = lambda enabled: torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=enabled)

    print("--- Beginning Physics-Informed Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)  # non_blocking pairs with pin_memory
            optimizer.zero_grad(set_to_none=True)         # set_to_none is faster than zeroing

            with autocast_ctx(use_amp):
                predictions = model(batch)
                loss = criterion(predictions, batch.y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with autocast_ctx(use_amp):
                    predictions = model(batch)
                    loss = criterion(predictions, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
                
        avg_val_loss = total_val_loss / len(val_dataset)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("Training Complete!")
    # OPTIMIZATION: save only state_dict of the underlying module if compiled
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(raw_model.state_dict(), "circledot_former_loopseq_weights.pth")
    print("Model weights saved to circledot_former_loopseq_weights.pth")

if __name__ == "__main__":
    MY_REAL_DATA_PATH = r"D:\exprement_16\data\41586_2020_3052_MOESM6_ESM.txt" 
    train_helical_model(MY_REAL_DATA_PATH, epochs=55, batch_size=128)