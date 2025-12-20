import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from training import train_lstm
from tools import custom_codecs

MODELS_DIR = PROJECT_ROOT / "models"


@torch.no_grad()
def generate_from_first(model, first_point, steps=200, device="cpu"):
    """
    first_point: Tensor[5] (already encoded/standardized like your training data)
    returns: Tensor[steps, 5] generated (includes first point as step 0)
    """
    model.eval()

    # first_point: Tensor[5]
    x_t = first_point.view(1, 1, 5).to(device)
    lengths = torch.tensor([1], dtype=torch.long, device=device)

    h = None
    out_seq = [first_point.cpu()]

    for _ in range(steps - 1):
        pred, h = model(x_t, lengths, h=h)     # <-- pass lengths explicitly
        next_point = pred[:, -1, :]            # [1,5]
        out_seq.append(next_point.squeeze(0).cpu())
        x_t = next_point.unsqueeze(1)          # [1,1,5]

    return torch.stack(out_seq, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
        "name": "h200_lr1e-3",
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0,
        "lr": 1e-3,
        "wd": 1e-4,
        "batch_size": 64,
        "epochs": 2000,
        "seed": 1
    }
loaded_model = train_lstm.NextStepLSTM(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
loaded_model.load_state_dict(torch.load(MODELS_DIR / 'nextstep_h200_l2_d0_best.pt', map_location=device))
loaded_model.to(device)

(x,y,vx,vy,fl) = custom_codecs.encode_flightpoint(97, 309, .12, 126, 1020)

current_point = torch.tensor([x, y, vx, vy, fl], dtype=torch.float32)

seq = generate_from_first(loaded_model, current_point, 350, device)
print(seq.shape)

for i in range(250):
    x1,y1,vx1,vy1,fl1 = seq[i].tolist()
    rho,theta,speed,heading,fl = custom_codecs.decode_flightpoint(x1, y1, vx1, vy1, fl1)
    #print(str(x)+","+str(y))
    print(fl)