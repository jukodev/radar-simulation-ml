"""
Plausibility Criteria Module

This module defines measurable plausibility criteria for flight trajectories
and calculates real-world boundaries from actual FlightPoints data.

Criteria include:
- Maximum jump distance per time step
- Speed range
- Climb/descent rate
- Heading change rate (turn rate)
- Rho range (radar distance)
- Flight level range
- Divergence detection
"""

import sqlite3
import sys
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "backup-31.12.2025.db"

# Radar update interval in seconds (approximately 5 seconds between updates)
RADAR_UPDATE_INTERVAL = 5.0


@dataclass
class PlausibilityCriteria:
    """
    Container for plausibility criteria with measurable limits.
    All values are calculated from real flight data.
    """
    # Jump distance (nautical miles per step)
    max_jump_distance: float = 0.0
    jump_distance_p99: float = 0.0
    jump_distance_p95: float = 0.0
    jump_distance_mean: float = 0.0
    jump_distance_std: float = 0.0
    
    # Speed (knots)
    speed_min: float = 0.0
    speed_max: float = 0.0
    speed_p01: float = 0.0
    speed_p99: float = 0.0
    speed_mean: float = 0.0
    speed_std: float = 0.0
    
    # Climb/descent rate (flight levels per step, ~FL/5sec)
    climb_rate_min: float = 0.0  # negative = descent
    climb_rate_max: float = 0.0
    climb_rate_p01: float = 0.0
    climb_rate_p99: float = 0.0
    climb_rate_mean: float = 0.0
    climb_rate_std: float = 0.0
    
    # Heading change rate / Turn rate (degrees per step)
    turn_rate_max: float = 0.0
    turn_rate_p99: float = 0.0
    turn_rate_p95: float = 0.0
    turn_rate_mean: float = 0.0
    turn_rate_std: float = 0.0
    
    # Rho range (nautical miles from radar)
    rho_min: float = 0.0
    rho_max: float = 0.0
    rho_p01: float = 0.0
    rho_p99: float = 0.0
    
    # Flight level range
    fl_min: float = 0.0
    fl_max: float = 0.0
    fl_p01: float = 0.0
    fl_p99: float = 0.0
    
    # Speed change rate (knots per step)
    speed_change_max: float = 0.0
    speed_change_p99: float = 0.0
    speed_change_mean: float = 0.0
    speed_change_std: float = 0.0
    
    # Number of data points analyzed
    num_points: int = 0
    num_transitions: int = 0
    num_trajectories: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "jump_distance": {
                "max": self.max_jump_distance,
                "p99": self.jump_distance_p99,
                "p95": self.jump_distance_p95,
                "mean": self.jump_distance_mean,
                "std": self.jump_distance_std,
            },
            "speed": {
                "min": self.speed_min,
                "max": self.speed_max,
                "p01": self.speed_p01,
                "p99": self.speed_p99,
                "mean": self.speed_mean,
                "std": self.speed_std,
            },
            "climb_rate": {
                "min": self.climb_rate_min,
                "max": self.climb_rate_max,
                "p01": self.climb_rate_p01,
                "p99": self.climb_rate_p99,
                "mean": self.climb_rate_mean,
                "std": self.climb_rate_std,
            },
            "turn_rate": {
                "max": self.turn_rate_max,
                "p99": self.turn_rate_p99,
                "p95": self.turn_rate_p95,
                "mean": self.turn_rate_mean,
                "std": self.turn_rate_std,
            },
            "rho": {
                "min": self.rho_min,
                "max": self.rho_max,
                "p01": self.rho_p01,
                "p99": self.rho_p99,
            },
            "flight_level": {
                "min": self.fl_min,
                "max": self.fl_max,
                "p01": self.fl_p01,
                "p99": self.fl_p99,
            },
            "speed_change": {
                "max": self.speed_change_max,
                "p99": self.speed_change_p99,
                "mean": self.speed_change_mean,
                "std": self.speed_change_std,
            },
            "data_stats": {
                "num_points": self.num_points,
                "num_transitions": self.num_transitions,
                "num_trajectories": self.num_trajectories,
            }
        }
    
    def print_report(self):
        """Print a formatted report of all criteria."""
        print("\n" + "=" * 70)
        print("PLAUSIBILITY CRITERIA - Real Data Boundaries")
        print("=" * 70)
        
        print(f"\n📊 Data analyzed: {self.num_trajectories:,} trajectories, "
              f"{self.num_points:,} points, {self.num_transitions:,} transitions")
        
        print("\n" + "-" * 70)
        print("1. JUMP DISTANCE (nautical miles per radar update)")
        print("-" * 70)
        print(f"   Max observed:      {self.max_jump_distance:.4f} NM")
        print(f"   99th percentile:   {self.jump_distance_p99:.4f} NM")
        print(f"   95th percentile:   {self.jump_distance_p95:.4f} NM")
        print(f"   Mean ± Std:        {self.jump_distance_mean:.4f} ± {self.jump_distance_std:.4f} NM")
        
        print("\n" + "-" * 70)
        print("2. SPEED (knots)")
        print("-" * 70)
        print(f"   Range:             [{self.speed_min:.1f}, {self.speed_max:.1f}] kn")
        print(f"   1st-99th %%:        [{self.speed_p01:.1f}, {self.speed_p99:.1f}] kn")
        print(f"   Mean ± Std:        {self.speed_mean:.1f} ± {self.speed_std:.1f} kn")
        
        print("\n" + "-" * 70)
        print("3. CLIMB/DESCENT RATE (FL per radar update, negative = descent)")
        print("-" * 70)
        print(f"   Range:             [{self.climb_rate_min:.2f}, {self.climb_rate_max:.2f}] FL/step")
        print(f"   1st-99th %%:        [{self.climb_rate_p01:.2f}, {self.climb_rate_p99:.2f}] FL/step")
        print(f"   Mean ± Std:        {self.climb_rate_mean:.4f} ± {self.climb_rate_std:.2f} FL/step")
        
        print("\n" + "-" * 70)
        print("4. TURN RATE (degrees per radar update, absolute)")
        print("-" * 70)
        print(f"   Max observed:      {self.turn_rate_max:.2f}°")
        print(f"   99th percentile:   {self.turn_rate_p99:.2f}°")
        print(f"   95th percentile:   {self.turn_rate_p95:.2f}°")
        print(f"   Mean ± Std:        {self.turn_rate_mean:.2f} ± {self.turn_rate_std:.2f}°")
        
        print("\n" + "-" * 70)
        print("5. RADAR DISTANCE / RHO (nautical miles)")
        print("-" * 70)
        print(f"   Range:             [{self.rho_min:.2f}, {self.rho_max:.2f}] NM")
        print(f"   1st-99th %%:        [{self.rho_p01:.2f}, {self.rho_p99:.2f}] NM")
        
        print("\n" + "-" * 70)
        print("6. FLIGHT LEVEL (FL, hundreds of feet)")
        print("-" * 70)
        print(f"   Range:             [{self.fl_min:.0f}, {self.fl_max:.0f}] FL")
        print(f"   1st-99th %%:        [{self.fl_p01:.0f}, {self.fl_p99:.0f}] FL")
        
        print("\n" + "-" * 70)
        print("7. SPEED CHANGE RATE (knots per radar update)")
        print("-" * 70)
        print(f"   Max observed:      {self.speed_change_max:.2f} kn/step")
        print(f"   99th percentile:   {self.speed_change_p99:.2f} kn/step")
        print(f"   Mean ± Std:        {self.speed_change_mean:.4f} ± {self.speed_change_std:.2f} kn/step")
        
        print("\n" + "=" * 70)


def calculate_heading_diff(h1: float, h2: float) -> float:
    """
    Calculate the signed heading difference between two headings.
    Returns value in range [-180, 180].
    """
    diff = h2 - h1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def calculate_distance(rho1: float, theta1: float, rho2: float, theta2: float) -> float:
    """
    Calculate the distance between two points in polar coordinates.
    Returns distance in nautical miles.
    """
    # Convert to Cartesian
    theta1_rad = math.radians(theta1)
    theta2_rad = math.radians(theta2)
    
    x1 = rho1 * math.cos(theta1_rad)
    y1 = rho1 * math.sin(theta1_rad)
    x2 = rho2 * math.cos(theta2_rad)
    y2 = rho2 * math.sin(theta2_rad)
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def load_trajectories_from_db(db_path: str, flight_type: int = 0) -> Dict[int, List[Tuple]]:
    """
    Load all trajectories from database, grouped by FlightId.
    
    :param db_path: Path to SQLite database
    :param flight_type: Type filter (0 = arrival, 1 = departure)
    :return: Dict mapping FlightId to list of (rho, theta, speed, heading, fl) tuples
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    query = """
    SELECT FlightId, Id, PositionRho, PositionTheta, VelocitySpeed, 
           VelocityHeading, FlightLevel
    FROM FlightPoints
    WHERE Type = ? 
      AND FlightId != 0
      AND PositionRho IS NOT NULL 
      AND PositionTheta IS NOT NULL
      AND VelocitySpeed IS NOT NULL
      AND VelocityHeading IS NOT NULL
      AND FlightLevel IS NOT NULL
    ORDER BY FlightId, Id
    """
    
    cur.execute(query, (flight_type,))
    rows = cur.fetchall()
    conn.close()
    
    # Group by FlightId
    trajectories: Dict[int, List[Tuple]] = {}
    for flight_id, point_id, rho, theta, speed, heading, fl in rows:
        if flight_id not in trajectories:
            trajectories[flight_id] = []
        trajectories[flight_id].append((rho, theta, speed, heading, fl))
    
    return trajectories


def calculate_criteria_from_db(db_path: str = None, flight_type: int = 0) -> PlausibilityCriteria:
    """
    Calculate plausibility criteria from real flight data in database.
    
    :param db_path: Path to SQLite database (defaults to DB_PATH)
    :param flight_type: Type filter (0 = arrival, 1 = departure)
    :return: PlausibilityCriteria with calculated values
    """
    if db_path is None:
        db_path = str(DB_PATH)
    
    print(f"Loading trajectories from {db_path}...")
    trajectories = load_trajectories_from_db(db_path, flight_type)
    
    if not trajectories:
        raise ValueError("No trajectories found in database!")
    
    # Collect all measurements
    all_rho = []
    all_speeds = []
    all_fl = []
    all_jump_distances = []
    all_climb_rates = []
    all_turn_rates = []
    all_speed_changes = []
    
    num_points = 0
    num_transitions = 0
    
    for flight_id, points in trajectories.items():
        for i, (rho, theta, speed, heading, fl) in enumerate(points):
            # Collect point-level metrics
            all_rho.append(rho)
            all_speeds.append(speed)
            all_fl.append(fl)
            num_points += 1
            
            # Collect transition-level metrics (comparing consecutive points)
            if i > 0:
                prev_rho, prev_theta, prev_speed, prev_heading, prev_fl = points[i - 1]
                
                # Jump distance
                jump_dist = calculate_distance(prev_rho, prev_theta, rho, theta)
                all_jump_distances.append(jump_dist)
                
                # Climb rate (FL change)
                climb_rate = fl - prev_fl
                all_climb_rates.append(climb_rate)
                
                # Turn rate (heading change, absolute)
                turn = abs(calculate_heading_diff(prev_heading, heading))
                all_turn_rates.append(turn)
                
                # Speed change
                speed_change = abs(speed - prev_speed)
                all_speed_changes.append(speed_change)
                
                num_transitions += 1
    
    # Convert to numpy arrays
    all_rho = np.array(all_rho)
    all_speeds = np.array(all_speeds)
    all_fl = np.array(all_fl)
    all_jump_distances = np.array(all_jump_distances)
    all_climb_rates = np.array(all_climb_rates)
    all_turn_rates = np.array(all_turn_rates)
    all_speed_changes = np.array(all_speed_changes)
    
    # Calculate criteria
    criteria = PlausibilityCriteria(
        # Jump distance
        max_jump_distance=float(np.max(all_jump_distances)),
        jump_distance_p99=float(np.percentile(all_jump_distances, 99)),
        jump_distance_p95=float(np.percentile(all_jump_distances, 95)),
        jump_distance_mean=float(np.mean(all_jump_distances)),
        jump_distance_std=float(np.std(all_jump_distances)),
        
        # Speed
        speed_min=float(np.min(all_speeds)),
        speed_max=float(np.max(all_speeds)),
        speed_p01=float(np.percentile(all_speeds, 1)),
        speed_p99=float(np.percentile(all_speeds, 99)),
        speed_mean=float(np.mean(all_speeds)),
        speed_std=float(np.std(all_speeds)),
        
        # Climb rate
        climb_rate_min=float(np.min(all_climb_rates)),
        climb_rate_max=float(np.max(all_climb_rates)),
        climb_rate_p01=float(np.percentile(all_climb_rates, 1)),
        climb_rate_p99=float(np.percentile(all_climb_rates, 99)),
        climb_rate_mean=float(np.mean(all_climb_rates)),
        climb_rate_std=float(np.std(all_climb_rates)),
        
        # Turn rate
        turn_rate_max=float(np.max(all_turn_rates)),
        turn_rate_p99=float(np.percentile(all_turn_rates, 99)),
        turn_rate_p95=float(np.percentile(all_turn_rates, 95)),
        turn_rate_mean=float(np.mean(all_turn_rates)),
        turn_rate_std=float(np.std(all_turn_rates)),
        
        # Rho
        rho_min=float(np.min(all_rho)),
        rho_max=float(np.max(all_rho)),
        rho_p01=float(np.percentile(all_rho, 1)),
        rho_p99=float(np.percentile(all_rho, 99)),
        
        # Flight level
        fl_min=float(np.min(all_fl)),
        fl_max=float(np.max(all_fl)),
        fl_p01=float(np.percentile(all_fl, 1)),
        fl_p99=float(np.percentile(all_fl, 99)),
        
        # Speed change
        speed_change_max=float(np.max(all_speed_changes)),
        speed_change_p99=float(np.percentile(all_speed_changes, 99)),
        speed_change_mean=float(np.mean(all_speed_changes)),
        speed_change_std=float(np.std(all_speed_changes)),
        
        # Stats
        num_points=num_points,
        num_transitions=num_transitions,
        num_trajectories=len(trajectories),
    )
    
    return criteria


@dataclass
class PlausibilityViolation:
    """Record of a plausibility violation."""
    criterion: str
    value: float
    limit: float
    step: int
    trajectory_id: int


@dataclass
class TrajectoryEvaluation:
    """Evaluation results for a single trajectory."""
    trajectory_id: int
    num_steps: int
    violations: List[PlausibilityViolation] = field(default_factory=list)
    
    # Per-criterion violation counts
    jump_distance_violations: int = 0
    speed_violations: int = 0
    climb_rate_violations: int = 0
    turn_rate_violations: int = 0
    rho_violations: int = 0
    fl_violations: int = 0
    speed_change_violations: int = 0
    
    # Divergence flag
    is_diverged: bool = False
    divergence_step: Optional[int] = None
    
    @property
    def total_violations(self) -> int:
        return len(self.violations)
    
    @property
    def is_plausible(self) -> bool:
        return self.total_violations == 0 and not self.is_diverged


def evaluate_trajectory(
    trajectory: List[Tuple[float, float, float, float, float]],
    criteria: PlausibilityCriteria,
    trajectory_id: int = 0,
    use_p99_limits: bool = True,
    divergence_threshold: float = 200.0,  # NM from origin
) -> TrajectoryEvaluation:
    """
    Evaluate a single trajectory against plausibility criteria.
    
    :param trajectory: List of (rho, theta, speed, heading, fl) tuples
    :param criteria: PlausibilityCriteria with limits
    :param trajectory_id: ID for tracking
    :param use_p99_limits: Use 99th percentile limits instead of max
    :param divergence_threshold: Rho threshold for divergence detection
    :return: TrajectoryEvaluation with results
    """
    eval_result = TrajectoryEvaluation(
        trajectory_id=trajectory_id,
        num_steps=len(trajectory),
    )
    
    # Select limits based on configuration
    jump_limit = criteria.jump_distance_p99 if use_p99_limits else criteria.max_jump_distance
    speed_min = criteria.speed_p01 if use_p99_limits else criteria.speed_min
    speed_max = criteria.speed_p99 if use_p99_limits else criteria.speed_max
    climb_min = criteria.climb_rate_p01 if use_p99_limits else criteria.climb_rate_min
    climb_max = criteria.climb_rate_p99 if use_p99_limits else criteria.climb_rate_max
    turn_limit = criteria.turn_rate_p99 if use_p99_limits else criteria.turn_rate_max
    rho_max = criteria.rho_p99 if use_p99_limits else criteria.rho_max
    fl_min = criteria.fl_p01 if use_p99_limits else criteria.fl_min
    fl_max = criteria.fl_p99 if use_p99_limits else criteria.fl_max
    speed_change_limit = criteria.speed_change_p99 if use_p99_limits else criteria.speed_change_max
    
    for i, (rho, theta, speed, heading, fl) in enumerate(trajectory):
        # Check for divergence (excessive distance from radar)
        if rho > divergence_threshold and not eval_result.is_diverged:
            eval_result.is_diverged = True
            eval_result.divergence_step = i
        
        # Check speed bounds
        if speed < speed_min or speed > speed_max:
            eval_result.violations.append(PlausibilityViolation(
                criterion="speed",
                value=speed,
                limit=speed_min if speed < speed_min else speed_max,
                step=i,
                trajectory_id=trajectory_id,
            ))
            eval_result.speed_violations += 1
        
        # Check rho bounds
        if rho > rho_max:
            eval_result.violations.append(PlausibilityViolation(
                criterion="rho",
                value=rho,
                limit=rho_max,
                step=i,
                trajectory_id=trajectory_id,
            ))
            eval_result.rho_violations += 1
        
        # Check flight level bounds
        if fl < fl_min or fl > fl_max:
            eval_result.violations.append(PlausibilityViolation(
                criterion="flight_level",
                value=fl,
                limit=fl_min if fl < fl_min else fl_max,
                step=i,
                trajectory_id=trajectory_id,
            ))
            eval_result.fl_violations += 1
        
        # Transition metrics (require previous point)
        if i > 0:
            prev_rho, prev_theta, prev_speed, prev_heading, prev_fl = trajectory[i - 1]
            
            # Jump distance
            jump_dist = calculate_distance(prev_rho, prev_theta, rho, theta)
            if jump_dist > jump_limit:
                eval_result.violations.append(PlausibilityViolation(
                    criterion="jump_distance",
                    value=jump_dist,
                    limit=jump_limit,
                    step=i,
                    trajectory_id=trajectory_id,
                ))
                eval_result.jump_distance_violations += 1
            
            # Climb rate
            climb_rate = fl - prev_fl
            if climb_rate < climb_min or climb_rate > climb_max:
                eval_result.violations.append(PlausibilityViolation(
                    criterion="climb_rate",
                    value=climb_rate,
                    limit=climb_min if climb_rate < climb_min else climb_max,
                    step=i,
                    trajectory_id=trajectory_id,
                ))
                eval_result.climb_rate_violations += 1
            
            # Turn rate
            turn = abs(calculate_heading_diff(prev_heading, heading))
            if turn > turn_limit:
                eval_result.violations.append(PlausibilityViolation(
                    criterion="turn_rate",
                    value=turn,
                    limit=turn_limit,
                    step=i,
                    trajectory_id=trajectory_id,
                ))
                eval_result.turn_rate_violations += 1
            
            # Speed change
            speed_change = abs(speed - prev_speed)
            if speed_change > speed_change_limit:
                eval_result.violations.append(PlausibilityViolation(
                    criterion="speed_change",
                    value=speed_change,
                    limit=speed_change_limit,
                    step=i,
                    trajectory_id=trajectory_id,
                ))
                eval_result.speed_change_violations += 1
    
    return eval_result


@dataclass
class ModelEvaluationSummary:
    """Summary of model evaluation across multiple trajectories."""
    model_name: str
    num_trajectories: int
    num_steps_per_trajectory: int
    
    # Per-criterion violation statistics
    jump_distance_violation_rate: float = 0.0
    speed_violation_rate: float = 0.0
    climb_rate_violation_rate: float = 0.0
    turn_rate_violation_rate: float = 0.0
    rho_violation_rate: float = 0.0
    fl_violation_rate: float = 0.0
    speed_change_violation_rate: float = 0.0
    
    # Overall statistics
    plausible_trajectory_rate: float = 0.0
    diverged_trajectory_rate: float = 0.0
    avg_violations_per_trajectory: float = 0.0
    
    # Divergence statistics
    avg_divergence_step: float = 0.0
    
    def print_report(self):
        """Print a formatted report."""
        print("\n" + "=" * 70)
        print(f"MODEL EVALUATION: {self.model_name}")
        print("=" * 70)
        print(f"Trajectories analyzed: {self.num_trajectories}")
        print(f"Steps per trajectory: {self.num_steps_per_trajectory}")
        
        print("\n" + "-" * 70)
        print("VIOLATION RATES (% of transitions with violations)")
        print("-" * 70)
        print(f"  Jump distance:      {self.jump_distance_violation_rate:6.2f}%")
        print(f"  Speed:              {self.speed_violation_rate:6.2f}%")
        print(f"  Climb rate:         {self.climb_rate_violation_rate:6.2f}%")
        print(f"  Turn rate:          {self.turn_rate_violation_rate:6.2f}%")
        print(f"  Rho (distance):     {self.rho_violation_rate:6.2f}%")
        print(f"  Flight level:       {self.fl_violation_rate:6.2f}%")
        print(f"  Speed change:       {self.speed_change_violation_rate:6.2f}%")
        
        print("\n" + "-" * 70)
        print("OVERALL QUALITY")
        print("-" * 70)
        print(f"  Plausible trajectories:     {self.plausible_trajectory_rate:6.2f}%")
        print(f"  Diverged trajectories:      {self.diverged_trajectory_rate:6.2f}%")
        print(f"  Avg violations/trajectory:  {self.avg_violations_per_trajectory:.2f}")
        if self.avg_divergence_step > 0:
            print(f"  Avg divergence step:        {self.avg_divergence_step:.1f}")
        
        print("=" * 70)


def summarize_evaluations(
    evaluations: List[TrajectoryEvaluation],
    model_name: str,
    num_steps: int,
) -> ModelEvaluationSummary:
    """
    Summarize evaluation results across multiple trajectories.
    
    :param evaluations: List of TrajectoryEvaluation results
    :param model_name: Name of the model being evaluated
    :param num_steps: Number of steps per trajectory
    :return: ModelEvaluationSummary with aggregated statistics
    """
    n_traj = len(evaluations)
    n_transitions = n_traj * (num_steps - 1)  # Each trajectory has n-1 transitions
    n_points = n_traj * num_steps
    
    total_jump_violations = sum(e.jump_distance_violations for e in evaluations)
    total_speed_violations = sum(e.speed_violations for e in evaluations)
    total_climb_violations = sum(e.climb_rate_violations for e in evaluations)
    total_turn_violations = sum(e.turn_rate_violations for e in evaluations)
    total_rho_violations = sum(e.rho_violations for e in evaluations)
    total_fl_violations = sum(e.fl_violations for e in evaluations)
    total_speed_change_violations = sum(e.speed_change_violations for e in evaluations)
    
    n_plausible = sum(1 for e in evaluations if e.is_plausible)
    n_diverged = sum(1 for e in evaluations if e.is_diverged)
    
    divergence_steps = [e.divergence_step for e in evaluations if e.divergence_step is not None]
    
    return ModelEvaluationSummary(
        model_name=model_name,
        num_trajectories=n_traj,
        num_steps_per_trajectory=num_steps,
        
        jump_distance_violation_rate=100 * total_jump_violations / n_transitions if n_transitions > 0 else 0,
        speed_violation_rate=100 * total_speed_violations / n_points if n_points > 0 else 0,
        climb_rate_violation_rate=100 * total_climb_violations / n_transitions if n_transitions > 0 else 0,
        turn_rate_violation_rate=100 * total_turn_violations / n_transitions if n_transitions > 0 else 0,
        rho_violation_rate=100 * total_rho_violations / n_points if n_points > 0 else 0,
        fl_violation_rate=100 * total_fl_violations / n_points if n_points > 0 else 0,
        speed_change_violation_rate=100 * total_speed_change_violations / n_transitions if n_transitions > 0 else 0,
        
        plausible_trajectory_rate=100 * n_plausible / n_traj if n_traj > 0 else 0,
        diverged_trajectory_rate=100 * n_diverged / n_traj if n_traj > 0 else 0,
        avg_violations_per_trajectory=sum(e.total_violations for e in evaluations) / n_traj if n_traj > 0 else 0,
        
        avg_divergence_step=np.mean(divergence_steps) if divergence_steps else 0,
    )


def main():
    """Calculate and display plausibility criteria from database."""
    print("Calculating plausibility criteria from real flight data...")
    
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please update DB_PATH to point to your database file.")
        return
    
    criteria = calculate_criteria_from_db(str(DB_PATH))
    criteria.print_report()
    
    # Save criteria as JSON
    import json
    output_path = PROJECT_ROOT / "analysis_out" / "plausibility_criteria.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(criteria.to_dict(), f, indent=2)
    
    print(f"\nCriteria saved to: {output_path}")


if __name__ == "__main__":
    main()
