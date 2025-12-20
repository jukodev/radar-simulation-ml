from fastapi import APIRouter, HTTPException, Request
from typing import List, Annotated
from src.api.models import FlightPoint
from src.tools import custom_codecs
import torch
import logging

logger = logging.getLogger("uvicorn.info")

router = APIRouter()


@torch.inference_mode()
def generate_from_first(model, first_point, steps=200, device="cpu"):
    model.eval()

    x_t = first_point.view(1, 1, 5).to(device)
    lengths = torch.tensor([1], dtype=torch.long, device=device)

    h = None
    out_seq = [first_point.cpu()]

    for _ in range(steps - 1):
        pred, h = model(x_t, lengths, h=h)
        next_point = pred[:, -1, :]          # [1,5]
        out_seq.append(next_point.squeeze(0).cpu())
        x_t = next_point.unsqueeze(1)        # [1,1,5]

    return torch.stack(out_seq, dim=0)

@router.put("/arrivals", response_model=List[FlightPoint], tags=["arrivals"])
def simulate_arrival(request: Request, flight_point: FlightPoint) -> List[FlightPoint]:
    try:
        logger.info(f"Simulating arrival for flight point: {flight_point}")
        model = request.app.state.model
        device = request.app.state.device

        # Encode incoming point (adapt to your FlightPoint schema!)
        x, y, vx, vy, fl = custom_codecs.encode_flightpoint(
            flight_point.rho,
            flight_point.theta,
            flight_point.speed,
            flight_point.heading,
            flight_point.fl,
        )

        current = torch.tensor([x, y, vx, vy, fl], dtype=torch.float32)

        seq = generate_from_first(model, current, steps=350, device=device)

        # Decode outputs to FlightPoint list
        simulated: List[FlightPoint] = []
        for i in range(seq.shape[0]):
            x1, y1, vx1, vy1, fl1 = seq[i].tolist()
            rho, theta, speed, heading, fl_dec = custom_codecs.decode_flightpoint(x1, y1, vx1, vy1, fl1)

            simulated.append(FlightPoint(
                rho=rho,
                theta=theta,
                speed=speed,
                heading=heading,
                fl=fl_dec
            ))

        return simulated

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))