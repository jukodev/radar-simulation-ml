"""
Plausibility Analysis Script

This script evaluates LSTM and MLP models against plausibility criteria,
analyzing how well generated trajectories conform to real-world flight dynamics.

Outputs include:
- Per-criterion violation rates
- Divergence statistics
- Comparative analysis between models
"""

import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_lstm import NextStepLSTM
from tools import custom_codecs
from analysis.plausibility_criteria import (
    PlausibilityCriteria,
    calculate_criteria_from_db,
    evaluate_trajectory,
    summarize_evaluations,
    TrajectoryEvaluation,
    ModelEvaluationSummary,
)

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "analysis_out"

# Database path
DB_PATH = DATA_DIR / "backup-31.12.2025.db"

# Model configurations
LSTM_CONFIGS = [
    {"name": "LSTM h600 l2 d0", "hidden_size": 600, "num_layers": 2, "dropout": 0,
     "path": MODELS_DIR / "nextstep_h600_l2_d0_best.pt"},
    {"name": "LSTM h600 l2 d0.1", "hidden_size": 600, "num_layers": 2, "dropout": 0.1,
     "path": MODELS_DIR / "nextstep_h600_l2_d0.1_best.pt"},
    {"name": "LSTM h400 l2 d0", "hidden_size": 400, "num_layers": 2, "dropout": 0,
     "path": MODELS_DIR / "nextstep_h400_l2_d0_best.pt"},
    {"name": "LSTM h400 l3 d0", "hidden_size": 400, "num_layers": 3, "dropout": 0,
     "path": MODELS_DIR / "nextstep_h400_l3_d0_best.pt"},
    {"name": "LSTM h200 l2 d0", "hidden_size": 200, "num_layers": 2, "dropout": 0,
     "path": MODELS_DIR / "nextstep_h200_l2_d0_best.pt"},
]

MLP_CONFIGS = [
    {"name": "MLP h128", "hidden_size": 128, 
     "path": MODELS_DIR / "movement_mlp_h128_best.pt"},
]

# Analysis parameters
NUM_TRAJECTORIES = 500
PREDICTION_STEPS = 200


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


def get_starting_points_from_db(db_path: str, num_points: int = 100) -> List[Tuple]:
    """
    Get starting points from the database.
    Selects points where Type=0 and uses the lowest ID per FlightId.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
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


@torch.no_grad()
def generate_lstm_trajectory(model, first_point: torch.Tensor, steps: int = 200, 
                              device: str = "cpu") -> List[Tuple[float, float, float, float, float]]:
    """
    Generate a trajectory using LSTM model.
    Returns list of decoded (rho, theta, speed, heading, fl) tuples.
    """
    model.eval()
    
    x_t = first_point.view(1, 1, 5).to(device)
    lengths = torch.tensor([1], dtype=torch.long, device=device)
    
    h = None
    trajectory = []
    
    # Decode and store first point
    x_std, y_std, vx_std, vy_std, fl_std = first_point.tolist()
    rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
        x_std, y_std, vx_std, vy_std, fl_std
    )
    trajectory.append((rho, theta, speed, heading, fl))
    
    for _ in range(steps - 1):
        pred, h = model(x_t, lengths, h=h)
        next_point = pred[:, -1, :]  # [1, 5]
        
        # Decode prediction
        x_std, y_std, vx_std, vy_std, fl_std = next_point.squeeze(0).cpu().tolist()
        rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
            x_std, y_std, vx_std, vy_std, fl_std
        )
        trajectory.append((rho, theta, speed, heading, fl))
        
        x_t = next_point.unsqueeze(1)  # [1, 1, 5]
    
    return trajectory


@torch.no_grad()
def generate_mlp_trajectory(model, first_point: torch.Tensor, steps: int = 200,
                            device: str = "cpu") -> List[Tuple[float, float, float, float, float]]:
    """
    Generate a trajectory using MLP model.
    Returns list of decoded (rho, theta, speed, heading, fl) tuples.
    """
    model.eval()
    
    point = first_point.view(1, 5).to(device)
    trajectory = []
    
    # Decode and store first point
    x_std, y_std, vx_std, vy_std, fl_std = first_point.tolist()
    rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
        x_std, y_std, vx_std, vy_std, fl_std
    )
    trajectory.append((rho, theta, speed, heading, fl))
    
    for _ in range(steps - 1):
        pred = model(point)  # [1, 5]
        
        # Decode prediction
        x_std, y_std, vx_std, vy_std, fl_std = pred.squeeze(0).cpu().tolist()
        rho, theta, speed, heading, fl = custom_codecs.decode_flightpoint(
            x_std, y_std, vx_std, vy_std, fl_std
        )
        trajectory.append((rho, theta, speed, heading, fl))
        
        point = pred
    
    return trajectory


def evaluate_model(
    model,
    model_name: str,
    model_type: str,  # "lstm" or "mlp"
    starting_points: List[Tuple],
    criteria: PlausibilityCriteria,
    num_steps: int,
    device: str = "cpu",
) -> ModelEvaluationSummary:
    """
    Evaluate a model by generating trajectories and checking plausibility.
    """
    print(f"\n  Evaluating {model_name}...")
    
    evaluations = []
    
    for i, (rho, theta, speed, heading, fl) in enumerate(starting_points):
        # Encode starting point
        encoded = custom_codecs.encode_flightpoint(rho, theta, speed, heading, fl)
        start_tensor = torch.tensor(encoded, dtype=torch.float32)
        
        # Generate trajectory
        if model_type == "lstm":
            trajectory = generate_lstm_trajectory(model, start_tensor, num_steps, device)
        else:
            trajectory = generate_mlp_trajectory(model, start_tensor, num_steps, device)
        
        # Evaluate trajectory
        eval_result = evaluate_trajectory(trajectory, criteria, trajectory_id=i)
        evaluations.append(eval_result)
        
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(starting_points)} trajectories")
    
    # Summarize results
    summary = summarize_evaluations(evaluations, model_name, num_steps)
    return summary


def create_comparison_chart(summaries: List[ModelEvaluationSummary], output_path: Path):
    """Create a bar chart comparing violation rates across models."""
    criteria_names = [
        "Jump Dist", "Speed", "Climb Rate", 
        "Turn Rate", "Rho", "FL", "Speed Δ"
    ]
    
    n_models = len(summaries)
    x = np.arange(len(criteria_names))
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, summary in enumerate(summaries):
        rates = [
            summary.jump_distance_violation_rate,
            summary.speed_violation_rate,
            summary.climb_rate_violation_rate,
            summary.turn_rate_violation_rate,
            summary.rho_violation_rate,
            summary.fl_violation_rate,
            summary.speed_change_violation_rate,
        ]
        offset = (i - n_models/2 + 0.5) * width
        rects = ax.bar(x + offset, rates, width, label=summary.model_name)
        
        # Add value labels on bars
        for rect, rate in zip(rects, rates):
            if rate > 0.5:  # Only label if significant
                ax.annotate(f'{rate:.1f}%',
                           xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Plausibility Criteria Violation Rates by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names)
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison chart saved to: {output_path}")
    plt.show()


def create_divergence_chart(summaries: List[ModelEvaluationSummary], output_path: Path):
    """Create a bar chart comparing divergence rates and plausibility."""
    model_names = [s.model_name for s in summaries]
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Divergence rates
    divergence_rates = [s.diverged_trajectory_rate for s in summaries]
    bars1 = ax1.bar(x, divergence_rates, width, color='coral', label='Diverged')
    ax1.set_ylabel('Divergence Rate (%)')
    ax1.set_title('Trajectory Divergence Rate by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylim(0, max(100, max(divergence_rates) * 1.2))
    
    for bar, rate in zip(bars1, divergence_rates):
        ax1.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plausibility rates (fully plausible trajectories)
    plausible_rates = [s.plausible_trajectory_rate for s in summaries]
    bars2 = ax2.bar(x, plausible_rates, width, color='seagreen', label='Plausible')
    ax2.set_ylabel('Plausible Trajectory Rate (%)')
    ax2.set_title('Fully Plausible Trajectories by Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    
    for bar, rate in zip(bars2, plausible_rates):
        ax2.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Divergence chart saved to: {output_path}")
    plt.show()


def create_violations_per_step_chart(
    model,
    model_name: str,
    model_type: str,
    starting_points: List[Tuple],
    criteria: PlausibilityCriteria,
    num_steps: int,
    device: str,
    output_path: Path,
):
    """Create a chart showing where violations occur across trajectory steps."""
    print(f"\n  Generating step-wise violation analysis for {model_name}...")
    
    # Count violations at each step
    step_violations = {
        "jump_distance": np.zeros(num_steps),
        "speed": np.zeros(num_steps),
        "climb_rate": np.zeros(num_steps),
        "turn_rate": np.zeros(num_steps),
        "rho": np.zeros(num_steps),
        "flight_level": np.zeros(num_steps),
        "speed_change": np.zeros(num_steps),
    }
    
    n_trajectories = len(starting_points)
    
    for i, (rho, theta, speed, heading, fl) in enumerate(starting_points):
        encoded = custom_codecs.encode_flightpoint(rho, theta, speed, heading, fl)
        start_tensor = torch.tensor(encoded, dtype=torch.float32)
        
        if model_type == "lstm":
            trajectory = generate_lstm_trajectory(model, start_tensor, num_steps, device)
        else:
            trajectory = generate_mlp_trajectory(model, start_tensor, num_steps, device)
        
        eval_result = evaluate_trajectory(trajectory, criteria, trajectory_id=i)
        
        for violation in eval_result.violations:
            if violation.step < num_steps:
                step_violations[violation.criterion][violation.step] += 1
    
    # Normalize to percentage
    for key in step_violations:
        step_violations[key] = 100 * step_violations[key] / n_trajectories
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    steps = np.arange(num_steps)
    
    # Stacked area chart
    labels = ["Jump Dist", "Speed", "Climb Rate", "Turn Rate", "Rho", "FL", "Speed Δ"]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    ax.stackplot(steps, 
                 step_violations["jump_distance"],
                 step_violations["speed"],
                 step_violations["climb_rate"],
                 step_violations["turn_rate"],
                 step_violations["rho"],
                 step_violations["flight_level"],
                 step_violations["speed_change"],
                 labels=labels, colors=colors, alpha=0.8)
    
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Violation Rate (% of trajectories)')
    ax.set_title(f'Violations per Step - {model_name}\n({n_trajectories} trajectories, {num_steps} steps each)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, num_steps - 1)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Step-wise chart saved to: {output_path}")
    plt.show()


def generate_detailed_report(
    criteria: PlausibilityCriteria,
    summaries: List[ModelEvaluationSummary],
    output_path: Path,
):
    """Generate a detailed Markdown report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Plausibility Analysis Report\n\n")
        f.write(f"Generated from {criteria.num_trajectories:,} real trajectories with "
                f"{criteria.num_points:,} data points.\n\n")
        
        f.write("## 1. Plausibility Criteria (from Real Data)\n\n")
        f.write("| Criterion | Min | Max | P1 | P99 | Mean | Std |\n")
        f.write("|-----------|-----|-----|----|----|------|-----|\n")
        
        crit = criteria.to_dict()
        
        f.write(f"| Jump Distance (NM) | - | {crit['jump_distance']['max']:.4f} | - | "
                f"{crit['jump_distance']['p99']:.4f} | {crit['jump_distance']['mean']:.4f} | "
                f"{crit['jump_distance']['std']:.4f} |\n")
        
        f.write(f"| Speed (kn) | {crit['speed']['min']:.1f} | {crit['speed']['max']:.1f} | "
                f"{crit['speed']['p01']:.1f} | {crit['speed']['p99']:.1f} | "
                f"{crit['speed']['mean']:.1f} | {crit['speed']['std']:.1f} |\n")
        
        f.write(f"| Climb Rate (FL/step) | {crit['climb_rate']['min']:.2f} | "
                f"{crit['climb_rate']['max']:.2f} | {crit['climb_rate']['p01']:.2f} | "
                f"{crit['climb_rate']['p99']:.2f} | {crit['climb_rate']['mean']:.4f} | "
                f"{crit['climb_rate']['std']:.2f} |\n")
        
        f.write(f"| Turn Rate (°/step) | - | {crit['turn_rate']['max']:.2f} | - | "
                f"{crit['turn_rate']['p99']:.2f} | {crit['turn_rate']['mean']:.2f} | "
                f"{crit['turn_rate']['std']:.2f} |\n")
        
        f.write(f"| Rho (NM) | {crit['rho']['min']:.2f} | {crit['rho']['max']:.2f} | "
                f"{crit['rho']['p01']:.2f} | {crit['rho']['p99']:.2f} | - | - |\n")
        
        f.write(f"| Flight Level | {crit['flight_level']['min']:.0f} | "
                f"{crit['flight_level']['max']:.0f} | {crit['flight_level']['p01']:.0f} | "
                f"{crit['flight_level']['p99']:.0f} | - | - |\n")
        
        f.write(f"| Speed Change (kn/step) | - | {crit['speed_change']['max']:.2f} | - | "
                f"{crit['speed_change']['p99']:.2f} | {crit['speed_change']['mean']:.4f} | "
                f"{crit['speed_change']['std']:.2f} |\n")
        
        f.write("\n## 2. Model Evaluation Summary\n\n")
        
        f.write("### Violation Rates (%)\n\n")
        f.write("| Model | Jump Dist | Speed | Climb | Turn | Rho | FL | Speed Δ |\n")
        f.write("|-------|-----------|-------|-------|------|-----|----|---------|\n")
        
        for s in summaries:
            f.write(f"| {s.model_name} | {s.jump_distance_violation_rate:.2f} | "
                    f"{s.speed_violation_rate:.2f} | {s.climb_rate_violation_rate:.2f} | "
                    f"{s.turn_rate_violation_rate:.2f} | {s.rho_violation_rate:.2f} | "
                    f"{s.fl_violation_rate:.2f} | {s.speed_change_violation_rate:.2f} |\n")
        
        f.write("\n### Overall Quality\n\n")
        f.write("| Model | Plausible % | Diverged % | Avg Violations |\n")
        f.write("|-------|-------------|------------|----------------|\n")
        
        for s in summaries:
            f.write(f"| {s.model_name} | {s.plausible_trajectory_rate:.2f} | "
                    f"{s.diverged_trajectory_rate:.2f} | {s.avg_violations_per_trajectory:.2f} |\n")
        
        f.write("\n## 3. Interpretation\n\n")
        f.write("- **Plausible %**: Percentage of trajectories with zero violations\n")
        f.write("- **Diverged %**: Percentage of trajectories that exceeded the radar range\n")
        f.write("- **Lower violation rates** indicate better conformance to real flight dynamics\n")
        f.write("- Limits used are **99th percentile** values from real data\n")
    
    print(f"\nDetailed report saved to: {output_path}")


def main():
    print("=" * 70)
    print("PLAUSIBILITY ANALYSIS")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check database
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return
    
    # Calculate plausibility criteria from real data
    print("\n" + "-" * 70)
    print("Step 1: Calculating plausibility criteria from real data...")
    print("-" * 70)
    
    criteria = calculate_criteria_from_db(str(DB_PATH))
    criteria.print_report()
    
    # Get starting points
    print("\n" + "-" * 70)
    print("Step 2: Loading starting points...")
    print("-" * 70)
    
    starting_points = get_starting_points_from_db(str(DB_PATH), NUM_TRAJECTORIES)
    print(f"Loaded {len(starting_points)} starting points")
    
    # Evaluate models
    print("\n" + "-" * 70)
    print("Step 3: Evaluating models...")
    print("-" * 70)
    
    summaries = []
    
    # Evaluate LSTM models
    for config in LSTM_CONFIGS:
        if not config["path"].exists():
            print(f"  Skipping {config['name']}: model file not found")
            continue
        
        model = NextStepLSTM(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
        model.load_state_dict(torch.load(config["path"], map_location=device))
        model.to(device)
        model.eval()
        
        summary = evaluate_model(
            model, config["name"], "lstm",
            starting_points, criteria, PREDICTION_STEPS, device
        )
        summary.print_report()
        summaries.append(summary)
    
    # Evaluate MLP models
    for config in MLP_CONFIGS:
        if not config["path"].exists():
            print(f"  Skipping {config['name']}: model file not found")
            continue
        
        model = MovementPredictor(hidden_size=config["hidden_size"])
        # Load state dict and handle torch.compile() prefix if present
        state_dict = torch.load(config["path"], map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[len("_orig_mod."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        summary = evaluate_model(
            model, config["name"], "mlp",
            starting_points, criteria, PREDICTION_STEPS, device
        )
        summary.print_report()
        summaries.append(summary)
    
    if not summaries:
        print("ERROR: No models were evaluated!")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate comparison charts
    print("\n" + "-" * 70)
    print("Step 4: Generating visualizations...")
    print("-" * 70)
    
    create_comparison_chart(summaries, OUTPUT_DIR / "plausibility_violation_comparison.png")
    create_divergence_chart(summaries, OUTPUT_DIR / "plausibility_divergence_comparison.png")
    
    # Generate step-wise analysis for best/worst models
    if summaries:
        # Find best LSTM model (lowest avg violations)
        lstm_summaries = [s for s in summaries if "LSTM" in s.model_name]
        if lstm_summaries:
            best_lstm = min(lstm_summaries, key=lambda s: s.avg_violations_per_trajectory)
            # Find corresponding config
            for config in LSTM_CONFIGS:
                if config["name"] == best_lstm.model_name:
                    model = NextStepLSTM(
                        hidden_size=config["hidden_size"],
                        num_layers=config["num_layers"],
                        dropout=config["dropout"]
                    )
                    model.load_state_dict(torch.load(config["path"], map_location=device))
                    model.to(device)
                    model.eval()
                    
                    create_violations_per_step_chart(
                        model, config["name"], "lstm",
                        starting_points[:100],  # Use subset for speed
                        criteria, PREDICTION_STEPS, device,
                        OUTPUT_DIR / f"plausibility_steps_{config['name'].replace(' ', '_')}.png"
                    )
                    break
    
    # Generate detailed report
    print("\n" + "-" * 70)
    print("Step 5: Generating report...")
    print("-" * 70)
    
    generate_detailed_report(
        criteria, summaries,
        OUTPUT_DIR / "plausibility_analysis_report.md"
    )
    
    # Save criteria as JSON
    import json
    with open(OUTPUT_DIR / "plausibility_criteria.json", "w") as f:
        json.dump(criteria.to_dict(), f, indent=2)
    print(f"Criteria saved to: {OUTPUT_DIR / 'plausibility_criteria.json'}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
