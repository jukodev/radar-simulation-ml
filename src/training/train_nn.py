import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class MovementDataset(Dataset):
    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu")
        self.X = data["X"].float()
        self.y = data["y"].float()

        if self.X.ndim != 2 or self.y.ndim != 2:
            raise ValueError(f"Expected X,y to be 2D tensors. Got X:{self.X.shape} y:{self.y.shape}")
        if self.X.size(0) != self.y.size(0):
            raise ValueError(f"X and y must have same first dim. Got X:{self.X.shape} y:{self.y.shape}")

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MovementPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def train_one(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True  # generally helps for fixed-size batches

    ds = MovementDataset(str(DATA_DIR / "flight_data.pt"))

    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, _test_ds = random_split(ds, [n_train, n_val, n_test], generator=g)

    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=pin,
    )

    model = MovementPredictor(
        input_size=cfg.get("input_size", 5),
        hidden_size=cfg.get("hidden_size", 128),
        output_size=cfg.get("output_size", 5),
    ).to(device)

    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_epoch = -1
    last_time = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                preds = model(batch_X)
                val_loss += criterion(preds, batch_y).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        duration = time.time() - last_time
        last_time = time.time()

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | LR: {current_lr:.2e} | "
            f"Time: {duration / 60:.2f}mins"
        )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODELS_DIR / f"movement_{cfg['name']}_best.pt")

        if epoch - best_epoch >= 20:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best was epoch {best_epoch} with val loss {best_val:.6f}."
            )
            break
    return cfg["name"], best_val


def run():
    cfg = {
        "name": "mlp_h128",
        "hidden_size": 128,
        "lr": 1e-3,
        "wd": 1e-4,
        "batch_size": 64,
        "epochs": 2000,
        "seed": 1,
        "num_workers": 4,
        # "input_size": 5,
        # "output_size": 5,
    }

    print("Starting " + cfg["name"])
    name, best = train_one(cfg)
    print("Best: " + str(best))


if __name__ == "__main__":
    run()
