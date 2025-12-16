import torch
import torch.nn as nn
import numpy as np
import custom_codecs
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

np.set_printoptions(suppress=True, precision=4)

def rollout_until_threshold(model, start_point, max_steps=1000):
    """
    Repeatedly predicts next point until max_steps is reached.

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
            pred_cpu = pred.detach().cpu()

            if pred_cpu.ndim == 2:
                pred_cpu = pred_cpu[0]

            x_std, y_std, vx_std, vy_std, fl_std = pred_cpu.tolist()
            print(str(x_std*100)+","+str(y_std*100))
            
            rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
                x_std, y_std, vx_std, vy_std, fl_std
            )
            #print(rho, theta, speed, heading, fl)
            
            trajectory.append(pred.cpu())

            point = pred


    return torch.cat(trajectory, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = MovementPredictor()
loaded_model.load_state_dict(torch.load('model.pt', map_location=device))
loaded_model.to(device)
loaded_model.eval()

(x,y,vx,vy,fl) = custom_codecs.encode_flightpoint(97, 309, .12, 133, 1020)

current_point = torch.tensor([[x, y, vx, vy, fl]], dtype=torch.float32)

trajectory = rollout_until_threshold(
    loaded_model,
    current_point,
    max_steps=150 
)

print("Trajectory shape:", trajectory.shape)
#print(trajectory.numpy())


