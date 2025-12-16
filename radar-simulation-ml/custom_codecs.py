import math
import statistics
from typing import Tuple

# Values saved from previous calculation, for set of 18861 trajectories
FL_MEAN = 473.93616315184147
FL_STDEV = 319.90238842157635

VX_MEAN = -0.002095183304311573
VX_STDEV = 0.0574749377907955

VY_MEAN = 0.00561096691598208
VY_STDEV = 0.06425699715580237

RHO_MAX = 100

def calulate_cartesian(rho: float, theta: float)-> Tuple[float, float]:
    """
    Calculates cartesian 2D coordinates for given polar coordinates.
    
    :param rho: Distance from origin
    :type rho: float
    :param theta: Angle in degrees cw from north
    :type theta: float
    :return: Cartesian x and y coordinates
    :rtype: Tuple[float, float]
    """
    angle_radian = theta * (math.pi / 180)
    x = rho * math.cos(angle_radian)
    y = rho * math.sin(angle_radian)
    return (x,y)

def calculate_radial(x: float, y: float) -> Tuple[float, float]:
    """
    Calculates polar 2D coordinates for given cartesian coordinates
    
    :param x: x coordinate
    :type x: float
    :param y: y coordinate
    :type y: float
    :return: Polar rho and theta (angle in degrees) coordinates. 
    :rtype: Tuple[float, float]
    """
    rho = math.sqrt(x**2 + y**2)          
    theta_radian = math.atan2(y, x)
    theta = theta_radian * 180 / math.pi
    return (rho, theta)

def encode_flightpoint(rho: float, theta: float, speed: float, heading: float, fl: int) -> Tuple[float, float, float, float, float]:
    """
    Normalizes and standardizes all values of the flightpoint.
    :return: Returns tuple of normalized and standardized floats.
    :rtype: Tuple[float, float, float, float, float]
    """
    (x,y) = calulate_cartesian(rho, theta)
    (vx, vy) = calulate_cartesian(speed, heading)

    x_std = x / RHO_MAX
    y_std = y / RHO_MAX

    vx_std = float(vx - VX_MEAN) / VX_STDEV
    vy_std = float(vy - VY_MEAN) / VY_STDEV

    fl_std = float(fl - FL_MEAN) / FL_STDEV

    return (x_std, y_std, vx_std, vy_std, fl_std)

def decode_flightpoint(
    x_std: float,
    y_std: float,
    vx_std: float,
    vy_std: float,
    fl_std: float,
) -> Tuple[float, float, float, float, int]:
    """
    Denormalizes and destandardizes the encoded flightpoint back to
    (rho, theta, speed, heading, fl).
    """
    x = x_std * RHO_MAX
    y = y_std * RHO_MAX

    vx = vx_std * VX_STDEV + VX_MEAN
    vy = vy_std * VY_STDEV + VY_MEAN

    fl = fl_std * FL_STDEV + FL_MEAN
    fl = int(round(fl))

    rho, theta = calculate_radial(x, y)
    speed, heading = calculate_radial(vx, vy)

    heading = (heading + 360.0) % 360.0

    return rho, theta, speed, heading, fl



def print_mean_and_stdev(rows: list[float]) -> None:
    """
    Calulates and prints mean and standard deviation for all relevant metrics.
    Run and update once when changing base dataset.
    :param rows: List of db flightpath entries
    :type rows: list[float]
    """
    fl_values = []
    vx_values = []
    vy_values = []

    for _, _, rho, theta, speed, heading, fl in rows:
        if None in (rho, theta, speed, heading, fl):
            continue

        heading_rad = math.radians(heading)
        x_vel = speed * math.cos(heading_rad)
        y_vel = speed * math.sin(heading_rad)

        fl_values.append(fl)
        vx_values.append(x_vel)
        vy_values.append(y_vel)

    fl_mean = statistics.mean(fl_values)
    fl_stdev = statistics.stdev(fl_values)

    vx_mean = statistics.mean(vx_values)
    vx_stdev = statistics.stdev(vx_values)

    vy_mean = statistics.mean(vy_values)
    vy_stdev = statistics.stdev(vy_values)

    print("fl mean:", fl_mean, "fl dev:", fl_stdev)
    print("vx mean:", vx_mean, "vx dev:", vx_stdev)
    print("vy mean:", vy_mean, "vy dev:", vy_stdev)
