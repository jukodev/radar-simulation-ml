from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

np.set_printoptions(suppress=True, precision=4)


class MovementPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.net(x)

def rollout_until_threshold(model, start_point, first_dim_threshold=5.0, max_steps=1000):
    """
    Repeatedly predicts next point until the first dimension < threshold
    or max_steps is reached.

    Returns: tensor of shape [T, 5] with the whole trajectory
             (including the starting point).
    """
    model.eval()
    device = next(model.parameters()).device

    point = start_point.to(device)
    if point.dim() == 1:
        point = point.unsqueeze(0)

    trajectory = [point.cpu()]  

    with torch.no_grad():
        for _ in range(max_steps):
            pred = model(point)    
            trajectory.append(pred.cpu())

            first_dim = pred[0, 4].item()
            print("fl: "+str(first_dim))
            if first_dim < first_dim_threshold:
                break

            point = pred

    return torch.cat(trajectory, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = MovementPredictor()
loaded_model.load_state_dict(torch.load(MODELS_DIR / 'model.pt', map_location=device))
loaded_model.to(device)
loaded_model.eval()

current_point = torch.tensor([[90.0, 350.0, 0.126, 150.0, 1010]], dtype=torch.float32)

trajectory = rollout_until_threshold(
    loaded_model,
    current_point,
    first_dim_threshold=5.0,
    max_steps=500  # just a safety cap
)

print("Trajectory shape:", trajectory.shape)
#print(trajectory.numpy())


