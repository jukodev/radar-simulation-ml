from pathlib import Path
import torch
from src.training import train_lstm

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

def load_model(device: torch.device) -> torch.nn.Module:
    cfg = {
        "hidden_size": 600,
        "num_layers": 2,
        "dropout": 0,
    }

    model = train_lstm.NextStepLSTM(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

    state = torch.load(MODELS_DIR / "nextstep_h600_l2_d0_best.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_nn_model(device: torch.device) -> torch.nn.Module:
    from src.training.train_nn import MovementPredictor

    model = MovementPredictor(
        input_size=5,
        hidden_size=128,
        output_size=5,
    )

    state = torch.load(MODELS_DIR / "movement_mlp_h128_best.pt", map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model