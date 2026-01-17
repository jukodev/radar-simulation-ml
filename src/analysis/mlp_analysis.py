"""
MLP Model Analysis Script

This script tests the MLP neural network model by sampling predictions and visualizing
them in position and altitude heatmaps.

Starting points are selected from the database where Type=0 and ID is lowest
in each flight sequence (i.e., the first point of each trajectory).
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools import custom_codecs

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "analysis_out"

# Database path - adjust as needed
DB_PATH = DATA_DIR / "backup-31.12.2025.db"

# Model configuration (adjust to match your trained model)
MODEL_CONFIG = {
    "hidden_size": 128,

}
MODEL_PATH = MODELS_DIR / "movement_mlp_h128_best.pt"

# Analysis parameters
NUM_PREDICTIONS = 1000
PREDICTION_STEPS = 200  # Number of steps to predict for each trajectory
BINS = 1000  # Heatmap resolution


class MovementPredictor(nn.Module):
    """MLP model for predicting next flight point."""
    def __init__(self, input_size=5, hidden_size=128, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def generate_from_first(model, first_point, steps=200, device="cpu"):
    """
    Generate a trajectory from a first point using the MLP model.
    
    :param model: Trained MovementPredictor model
    :param first_point: Tensor[5] (encoded/standardized)
    :param steps: Number of steps to generate
    :param device: Device to run on
    :return: Tensor[steps, 5] generated trajectory
    """
    model.eval()

    point = first_point.view(1, 5).to(device)
    out_seq = [first_point.cpu()]

    for _ in range(steps - 1):
        pred = model(point)  # [1, 5]
        out_seq.append(pred.squeeze(0).cpu())
        point = pred

    return torch.stack(out_seq, dim=0)


def get_starting_points_from_db(db_path: str, num_points: int = 100):
    """
    Get starting points from the database.
    Selects points where Type=0 and uses the lowest ID per FlightId
    (i.e., the first point of each trajectory).
    
    :param db_path: Path to SQLite database
    :param num_points: Number of starting points to retrieve
    :return: List of tuples (rho, theta, speed, heading, fl)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get the first point (lowest ID) for each flight where Type=0
    query = """
    SELECT fp.PositionRho, fp.PositionTheta, fp.VelocitySpeed, 
           fp.VelocityHeading, fp.FlightLevel
    FROM FlightPoints fp
    INNER JOIN (
        SELECT FlightId, MIN(Id) as MinId
        FROM FlightPoints
        WHERE Type = 0 AND FlightId != 0
        GROUP BY FlightId
    ) first_points ON fp.Id = first_points.MinId
    WHERE fp.PositionRho IS NOT NULL 
      AND fp.PositionTheta IS NOT NULL
      AND fp.VelocitySpeed IS NOT NULL
      AND fp.VelocityHeading IS NOT NULL
      AND fp.FlightLevel IS NOT NULL
    ORDER BY RANDOM()
    LIMIT ?
    """
    
    cur.execute(query, (num_points,))
    rows = cur.fetchall()
    conn.close()
    
    return rows


def generate_random_outer_starting_points(num_points: int = 100):
    """
    Generate random starting points on the outer edge of the radar coverage.
    
    :param num_points: Number of starting points to generate
    :return: List of tuples (rho, theta, speed, heading, fl)
    """
    starting_points = []
    for _ in range(num_points):
        rho = 100
        theta = np.random.uniform(0, 360)
        speed = 0.1
        heading = (theta - 180) % 360  # Pointing inward toward center
        fl = 1000
        starting_points.append((rho, theta, speed, heading, fl))
    return starting_points


def decode_trajectory(trajectory: torch.Tensor):
    """
    Decode a full trajectory from standardized format to original values.
    
    :param trajectory: Tensor[T, 5] in standardized format
    :return: Tuple of numpy arrays (x_positions, y_positions, flight_levels)
    """
    x_positions = []
    y_positions = []
    flight_levels = []
    
    for i in range(trajectory.shape[0]):
        x_std, y_std, vx_std, vy_std, fl_std = trajectory[i].tolist()
        rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
            x_std, y_std, vx_std, vy_std, fl_std
        )
        
        # Convert polar to Cartesian for position
        theta_rad = np.deg2rad(theta)
        x = rho * np.sin(theta_rad)
        y = rho * np.cos(theta_rad)
        
        x_positions.append(x)
        y_positions.append(y)
        flight_levels.append(fl)
    
    return np.array(x_positions), np.array(y_positions), np.array(flight_levels)


def create_position_heatmap(all_x: np.ndarray, all_y: np.ndarray, 
                            output_path: Path = None, bins: int = BINS,
                            title_prefix: str = "MLP",
                            xy_range: tuple = (-100, 100)):
    """
    Create a 2D position heatmap.
    
    :param all_x: Array of all x positions
    :param all_y: Array of all y positions
    :param output_path: Optional path to save the figure
    :param bins: Number of bins for histogram
    :param title_prefix: Prefix for the title
    :param xy_range: Tuple of (min, max) for x and y axes
    """
    # Create 2D histogram with fixed range
    H, x_edges, y_edges = np.histogram2d(
        all_x, all_y, bins=bins,
        range=[[xy_range[0], xy_range[1]], [xy_range[0], xy_range[1]]]
    )
    
    # Apply log scale for better visualization
    H_log = np.log1p(H)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(
        H_log.T,
        origin="lower",
        extent=[xy_range[0], xy_range[1], xy_range[0], xy_range[1]],
        aspect="equal",
        cmap="inferno",
    )
    plt.colorbar(label="log(1 + count)")
    plt.title(f"{title_prefix} Predicted Position Heatmap\n({NUM_PREDICTIONS} trajectories, {PREDICTION_STEPS} steps each)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Position heatmap saved to: {output_path}")
    
    plt.show()


def create_altitude_heatmap(all_time: np.ndarray, all_fl: np.ndarray, 
                            output_path: Path = None, bins: tuple = (PREDICTION_STEPS, 1000),
                            title_prefix: str = "MLP",
                            fl_range: tuple = (0, 1100)):
    """
    Create an altitude heatmap showing flight level (y) vs step index (x).
    
    :param all_time: Array of all step indices
    :param all_fl: Array of all flight levels
    :param output_path: Optional path to save the figure
    :param bins: Tuple of (time_bins, fl_bins) for histogram
    :param title_prefix: Prefix for the title
    :param fl_range: Tuple of (min, max) for flight level axis
    """
    # Create 2D histogram with time on x-axis and altitude on y-axis
    H, time_edges, fl_edges = np.histogram2d(
        all_time, all_fl, bins=bins,
        range=[[0, PREDICTION_STEPS], [fl_range[0], fl_range[1]]]
    )
    
    # Apply log scale for better visualization
    H_log = np.log1p(H)
    
    plt.figure(figsize=(14, 8))
    im = plt.imshow(
        H_log.T,
        origin="lower",
        extent=[0, PREDICTION_STEPS, fl_range[0], fl_range[1]],
        aspect="auto",
        cmap="inferno",
        interpolation="gaussian",
    )
    plt.colorbar(im, label="log(1 + count)")
    plt.title(f"{title_prefix} Predicted Altitude over Time\n({NUM_PREDICTIONS} trajectories, {PREDICTION_STEPS} steps each)")
    plt.xlabel("Step Index")
    plt.ylabel("Flight Level (FL)")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Altitude heatmap saved to: {output_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("MLP Model Analysis")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = MovementPredictor(
        hidden_size=MODEL_CONFIG["hidden_size"],
    )
    
    # Load state dict and handle torch.compile() prefix if present
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # Remove "_orig_mod." prefix if model was saved with torch.compile()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get starting points from database
    print(f"\nLoading {NUM_PREDICTIONS} starting points from database...")
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please update DB_PATH to point to your database file.")
        return
    
    starting_points = get_starting_points_from_db(str(DB_PATH), NUM_PREDICTIONS)
    print(f"Loaded {len(starting_points)} starting points")
    
    if len(starting_points) == 0:
        print("ERROR: No starting points found in database.")
        return
    
    # Generate predictions
    print(f"\nGenerating {PREDICTION_STEPS}-step predictions for each trajectory...")
    all_x = []
    all_y = []
    all_fl = []
    all_time = []
    
    for i, (rho, theta, speed, heading, fl) in enumerate(starting_points):
        # Encode the starting point
        encoded = custom_codecs.encode_flightpoint(rho, theta, speed, heading, fl)
        start_tensor = torch.tensor(encoded, dtype=torch.float32)
        
        # Generate trajectory
        trajectory = generate_from_first(model, start_tensor, PREDICTION_STEPS, device)
        
        # Decode trajectory
        x_pos, y_pos, flight_levels = decode_trajectory(trajectory)
        
        # Calculate time values (packet index)
        time_values = list(range(len(flight_levels)))
        
        all_x.extend(x_pos)
        all_y.extend(y_pos)
        all_fl.extend(flight_levels)
        all_time.extend(time_values)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(starting_points)} trajectories...")
    
    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_fl = np.array(all_fl)
    all_time = np.array(all_time)
    
    print(f"\nTotal points generated: {len(all_x)}")
    print(f"Position range: X=[{all_x.min():.1f}, {all_x.max():.1f}], "
          f"Y=[{all_y.min():.1f}, {all_y.max():.1f}]")
    print(f"Flight level range: [{all_fl.min()}, {all_fl.max()}]")
    print(f"Step index range: [{all_time.min()}, {all_time.max()}]")
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create heatmaps
    print("\nCreating position heatmap...")
    create_position_heatmap(
        all_x, all_y,
        output_path=OUTPUT_DIR / "mlp_position_heatmap.png"
    )
    
    print("\nCreating altitude heatmap...")
    create_altitude_heatmap(
        all_time, all_fl,
        output_path=OUTPUT_DIR / "mlp_altitude_heatmap.png"
    )
    
    # Generate predictions from random outer positions
    print("\n" + "=" * 60)
    print("Generating predictions from random outer positions...")
    print("=" * 60)
    
    outer_starting_points = generate_random_outer_starting_points(NUM_PREDICTIONS)
    print(f"Generated {len(outer_starting_points)} random outer starting points")
    
    outer_x = []
    outer_y = []
    
    for i, (rho, theta, speed, heading, fl) in enumerate(outer_starting_points):
        # Encode the starting point
        encoded = custom_codecs.encode_flightpoint(rho, theta, speed, heading, fl)
        start_tensor = torch.tensor(encoded, dtype=torch.float32)
        
        # Generate trajectory
        trajectory = generate_from_first(model, start_tensor, PREDICTION_STEPS, device)
        
        # Decode trajectory
        x_pos, y_pos, _ = decode_trajectory(trajectory)
        
        outer_x.extend(x_pos)
        outer_y.extend(y_pos)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(outer_starting_points)} trajectories...")
    
    outer_x = np.array(outer_x)
    outer_y = np.array(outer_y)
    
    print(f"\nTotal points generated: {len(outer_x)}")
    print(f"Position range: X=[{outer_x.min():.1f}, {outer_x.max():.1f}], "
          f"Y=[{outer_y.min():.1f}, {outer_y.max():.1f}]")
    
    print("\nCreating random outer position heatmap...")
    create_position_heatmap(
        outer_x, outer_y,
        output_path=OUTPUT_DIR / "mlp_random_outer_position_heatmap.png",
        title_prefix="MLP (Random Outer Start)"
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
