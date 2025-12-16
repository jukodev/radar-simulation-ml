import torch

data = torch.load("flight_data.pt")

X = data["X"]
y = data["y"]

N = 12577

# Simple split
train_ratio = 0.8
n_train = int(N * train_ratio)

X_train = X[:n_train]
y_train = y[:n_train]

X_val = X[n_train:]
y_val = y[n_train:]

from torch.utils.data import Dataset, DataLoader

class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MovementDataset(X_train, y_train)
val_dataset   = MovementDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch.nn as nn

class MovementPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)

model = MovementPredictor()

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # --- Forward pass
        preds = model(batch_X)          # shape: [batch_size, 2]

        # --- Compute loss
        loss = criterion(preds, batch_y)

        # --- Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_X.size(0)

    avg_train_loss = total_train_loss / len(train_dataset)

    # --- Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            total_val_loss += loss.item() * batch_X.size(0)

    avg_val_loss = total_val_loss / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"- Train Loss: {avg_train_loss:.4f} "
          f"- Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), 'model.pt')