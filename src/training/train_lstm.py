import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu")
        self.trajs = [t.float() for t in data["trajectories"] if t.size(0) >= 2]

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]          # [T, 5]
        x = traj[:-1]                   # [T-1, 5]
        y = traj[1:]                    # [T-1, 5]
        return x, y


def collate_padded(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)

    x_pad = pad_sequence(xs, batch_first=True)        # [B, Lmax, 5]
    y_pad = pad_sequence(ys, batch_first=True)        # [B, Lmax, 5]

    Lmax = x_pad.size(1)
    mask = (torch.arange(Lmax).unsqueeze(0) < lengths.unsqueeze(1))  # [B, Lmax]
    return x_pad, y_pad, mask, lengths


class NextStepLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x, lengths, h=None):
        # lengths must be on CPU for pack_padded_sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.lstm(packed, h)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        pred = self.head(out)
        return pred, h


def masked_mse(pred, target, mask):
    mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask_f
    return diff2.sum() / (mask_f.sum().clamp(min=1.0) * pred.size(-1))


def train_one(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    ds = TrajectoryDataset(DATA_DIR / "flight_data.pt")

    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

    # For GPU training: pin_memory helps host->device copies.
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_padded,
        num_workers=4,          # you can tune this
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_padded,
        num_workers=4,
        pin_memory=pin,
    )

    model = NextStepLSTM(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_val = float("inf")
    best_epoch = -1
    last_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0

        for x_pad, y_pad, mask, lengths in train_loader:
            x_pad = x_pad.to(device, non_blocking=True)
            y_pad = y_pad.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred, _ = model(x_pad, lengths)
            loss = masked_mse(pred, y_pad, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_pad, y_pad, mask, lengths in val_loader:
                x_pad = x_pad.to(device, non_blocking=True)
                y_pad = y_pad.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                pred, _ = model(x_pad, lengths)
                val_loss += masked_mse(pred, y_pad, mask).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        duration = time.time() - last_time
        last_time = time.time()

        current_lr = opt.param_groups[0]['lr']
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | LR: {current_lr:.2e} | "
            f"Time: {duration / 60:.2f}mins"
        )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / f"nextstep_{cfg['name']}_best.pt")
            best_epoch = epoch

        if epoch - best_epoch >= 20:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best was epoch {best_epoch} with val loss {best_val:.6f}."
            )
            break

    return cfg["name"], best_val


def run_sweep():
    cfg = {
        "name": "h200_l4_d0",
        "hidden_size": 200,
        "num_layers": 4,
        "dropout": 0,
        "lr": 1e-3,
        "wd": 1e-4,
        "batch_size": 64,
        "epochs": 2000,
        "seed": 1
    }

    print("Starting " + cfg["name"])
    name, best = train_one(cfg)
    print("Best: " + str(best))


if __name__ == "__main__":
    run_sweep()
