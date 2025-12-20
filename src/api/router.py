from fastapi import APIRouter
from .endpoints import simulation_routes
import logging

logger = logging.getLogger("uvicorn.info")

router = APIRouter()

router.include_router(simulation_routes, prefix="/simulations", tags=["simulations"])

@router.get("/", tags=["root"])
async def read_root() -> dict[str, str]:
    return {"message": "welcome to radar simulation ml api"}