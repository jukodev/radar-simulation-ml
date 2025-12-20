from pydantic import Field, BaseModel
from typing import List

class FlightPoint(BaseModel):
    rho: float = Field(..., description="Radial distance from radar in nautical miles")
    theta: float = Field(..., description="Angle from north in degrees")
    speed: float = Field(..., description="Speed in knots")
    heading: float = Field(..., description="Heading in degrees")
    fl: float = Field(..., description="Flight level in hundreds of feet")
