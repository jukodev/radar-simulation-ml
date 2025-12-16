
import torch
import train_lstm
import custom_codecs



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

cfg = {"name": "h128_lr1e-3", "hidden_size": 128, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 1e-4,
         "batch_size": 64, "epochs": 30, "threads": 2, "seed": 1}
loaded_model = train_lstm.NextStepLSTM(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"])
loaded_model.load_state_dict(torch.load('nextstep_h128_lr1e-3_best.pt', map_location=device))
loaded_model.to(device)

(x,y,vx,vy,fl) = custom_codecs.encode_flightpoint(97, 309, .12, 0, 1020)

current_point = torch.tensor([x, y, vx, vy, fl], dtype=torch.float32)

seq = generate_from_first(loaded_model, current_point, 250, device)
print(seq.shape)

for i in range(250):
    x,y,vx,vy,fl = seq[i].tolist()
    print(str(x*100)+","+str(y*100))