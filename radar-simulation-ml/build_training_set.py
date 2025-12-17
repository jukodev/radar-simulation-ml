"""
Extract flight trajectories from a SQLite database and save them as a PyTorch .pt file.

- Each trajectory consists of all points with the same FlightId
  ordered by Time.
- Each point has features:
    (PositionRho, PositionTheta, VelocitySpeed, VelocityHeading, FlightLevel)
- Additionally, we build (X, y) pairs:
    X = current point, y = next point in the same trajectory.

Result is saved as flight_data.pt
"""

import sqlite3
import math
import statistics
from typing import Dict, List, Tuple
import custom_codecs

import torch

DB_PATH = "backup-17.12.2025.db"
SAVE_PATH = "flight_data.pt"   
MIN_POINTS_PER_TRAJ = 50           

def load_trajectories_from_db(
    db_path: str,
    min_points_per_traj: int = 2
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Load trajectories from the SQLite database.

    Returns:
        flight_ids:   list of FlightId (int)
        trajectories: list of tensors, each of shape [T_i, 5]
                      with columns (rho, theta, speed, heading, flightlevel)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = """
    SELECT FlightId,
           Time,
           PositionRho,
           PositionTheta,
           VelocitySpeed,
           VelocityHeading,
           FlightLevel
    FROM FlightPoints
    WHERE FlightId != 0 AND Type = 0
    ORDER BY FlightId, Time;
    """

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    trajectories_raw: Dict[int, List[Tuple[float, float, float, float, float]]] = {}

    custom_codecs.print_mean_and_stdev(rows);

    for flight_id, time, rho, theta, speed, heading, fl in rows:
        if None in (rho, theta, speed, heading, fl):
            continue


        features = custom_codecs.encode_flightpoint(rho, theta, speed, heading, fl)

        if flight_id not in trajectories_raw:
            trajectories_raw[flight_id] = []
        trajectories_raw[flight_id].append(features)

    # Convert to tensors and filter by length
    flight_ids: List[int] = []
    trajectories: List[torch.Tensor] = []

    for flight_id, points in trajectories_raw.items():
        if len(points) < min_points_per_traj:
            # Too short to be useful - shouldnt exist in first place but to be sure
            continue

        traj_tensor = torch.tensor(points, dtype=torch.float32)  # shape [T, 5]
        flight_ids.append(flight_id)
        trajectories.append(traj_tensor)

    return flight_ids, trajectories


def build_xy_from_trajectories(
    trajectories: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From a list of trajectories, build (X, y) pairs:

    For each trajectory with points p[0], ..., p[T-1]:
      we add pairs (p[t], p[t+1]) for t = 0..T-2.

    Also returns:
      trajectory_index: which trajectory each pair came from
      time_index:       the time index within that trajectory (t for p[t] -> p[t+1])
    """
    X_list = []
    y_list = []
    traj_idx_list = []
    time_idx_list = []

    for i, traj in enumerate(trajectories):
        # traj: [T, 5]
        if traj.size(0) < 2:
            continue

        curr = traj[:-1, :]  # [T-1, 5]
        nxt = traj[1:, :]    # [T-1, 5]

        X_list.append(curr)
        y_list.append(nxt)

        # Keep track of which trajectory each pair belongs to
        T_minus_1 = curr.size(0)
        traj_idx_list.append(torch.full((T_minus_1,), i, dtype=torch.long))
        time_idx_list.append(torch.arange(T_minus_1, dtype=torch.long))

    X = torch.cat(X_list, dim=0)  # [N, 5]
    y = torch.cat(y_list, dim=0)  # [N, 5]
    trajectory_index = torch.cat(traj_idx_list, dim=0)  # [N]
    time_index = torch.cat(time_idx_list, dim=0)        # [N]


    return X, y, trajectory_index, time_index


def main():
    print("Loading trajectories from DB...")
    flight_ids, trajectories = load_trajectories_from_db(DB_PATH, MIN_POINTS_PER_TRAJ)

    print(f"Loaded {len(trajectories)} trajectories")

    # Build (X, y) for next-step prediction
    print("Building (X, y) pairs...")
    X, y, trajectory_index, time_index = build_xy_from_trajectories(trajectories)

    print("Shapes:")
    print("  #trajectories:", len(trajectories))
    print("  X:", X.shape)  # [N, 5]
    print("  y:", y.shape)  # [N, 5]

    data = {
        "trajectories": trajectories,          # List[Tensor[T_i, 5]]
        "flight_ids": flight_ids,              # List[int]
        "X": X,                                # [N, 5] current point
        "y": y,                                # [N, 5] next point
        "trajectory_index": trajectory_index,  # [N] index into trajectories list
        "time_index": time_index,              # [N] time step in that trajectory
    }

    torch.save(data, SAVE_PATH)
    print(f"Saved dataset to {SAVE_PATH}")


if __name__ == "__main__":
    main()
