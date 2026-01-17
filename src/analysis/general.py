#!/usr/bin/env python3
import os
import math
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DB_PATH = "C:\\Projekte\\radar-simulation-validator\\RadarSimulationValidator\\backup-31.12.2025.db"

OUT_DIR = "analysis_out"

CHUNK_ROWS = 500_000

# Heatmaps
BINS_POS = 800
MIN_COUNT_PER_BIN_MEAN_ALT = 10

# Time delta histogram (seconds) bins: 1s .. 2h
DT_BINS_S = np.logspace(0, np.log10(7200), 220)

# Sampling for percentiles (approx): keep about ~SAMPLE_TARGET rows using Id%step==0
SAMPLE_TARGET = 1_000_000

# Angle convention
THETA_IS_DEGREES = True

# If you want to filter, add SQL here (keep leading space).
# Example: FILTER_SQL = " AND Status = 1"
FILTER_SQL = ""


# =========================
# HELPERS
# =========================
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    # Some pragmas for read-heavy workloads
    con.execute("PRAGMA journal_mode=OFF;")
    con.execute("PRAGMA synchronous=OFF;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


def polar_to_xy_azimuth(rho: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Azimuth: 0 = North, clockwise positive
    if THETA_IS_DEGREES:
        theta = np.deg2rad(theta)
    x = rho * np.sin(theta)  # East
    y = rho * np.cos(theta)  # North
    return x, y


def fetch_one(con: sqlite3.Connection, sql: str, params=()):
    cur = con.execute(sql, params)
    return cur.fetchone()


def fetch_all(con: sqlite3.Connection, sql: str, params=()):
    cur = con.execute(sql, params)
    return cur.fetchall()


def iter_id_ranges(con: sqlite3.Connection, chunk: int):
    min_id, max_id = fetch_one(con, "SELECT MIN(Id), MAX(Id) FROM AsterixPackets")
    if min_id is None:
        return
    start = int(min_id)
    max_id = int(max_id)
    while start <= max_id:
        end = start + chunk - 1
        yield start, end
        start = end + 1


def iter_rows_positions_alt(con: sqlite3.Connection, chunk: int):
    # Stream PositionRho, PositionTheta, FlightLevelFeet
    for start, end in iter_id_ranges(con, chunk):
        sql = f"""
            SELECT PositionRho, PositionTheta, FlightLevelFeet
            FROM AsterixPackets
            WHERE Id BETWEEN ? AND ? {FILTER_SQL}
        """
        rows = con.execute(sql, (start, end)).fetchall()
        if rows:
            yield np.asarray(rows, dtype=np.float64)


def iter_rows_speeds_headings_fls(con: sqlite3.Connection, chunk: int):
    # Stream VelocitySpeed, VelocityHeading, FlightLevelFeet
    for start, end in iter_id_ranges(con, chunk):
        sql = f"""
            SELECT VelocitySpeed, VelocityHeading, FlightLevelFeet
            FROM AsterixPackets
            WHERE Id BETWEEN ? AND ? {FILTER_SQL}
        """
        rows = con.execute(sql, (start, end)).fetchall()
        if rows:
            yield np.asarray(rows, dtype=np.float64)


def approx_sample_step(con: sqlite3.Connection, target: int) -> int:
    max_id = fetch_one(con, "SELECT MAX(Id) FROM AsterixPackets")[0]
    if not max_id:
        return 1
    max_id = int(max_id)
    step = max(1, max_id // max(1, target))
    return step


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================
# OVERVIEW / PRINT STATS
# =========================
def print_overview(con: sqlite3.Connection):
    print_section("BASIC OVERVIEW")

    total = fetch_one(con, f"SELECT COUNT(*) FROM AsterixPackets WHERE 1=1 {FILTER_SQL}")[0]
    print(f"Rows: {total:,}")

    # Time range (ISO string + julianday as numeric)
    tmin, tmax = fetch_one(
        con,
        f"SELECT MIN(TimeOfDay), MAX(TimeOfDay) FROM AsterixPackets WHERE TimeOfDay IS NOT NULL {FILTER_SQL}",
    )
    print(f"TimeOfDay range (string): {tmin}  ->  {tmax}")

    jdmin, jdmax = fetch_one(
        con,
        f"SELECT MIN(julianday(TimeOfDay)), MAX(julianday(TimeOfDay)) FROM AsterixPackets WHERE TimeOfDay IS NOT NULL {FILTER_SQL}",
    )
    if jdmin is not None and jdmax is not None:
        span_days = float(jdmax) - float(jdmin)
        print(f"Time span: {span_days:.4f} days (~{span_days * 24:.2f} hours)")

    uniq_aircraft = fetch_one(
        con,
        f"SELECT COUNT(DISTINCT AircraftAddress) FROM AsterixPackets WHERE AircraftAddress IS NOT NULL {FILTER_SQL}",
    )[0]
    print(f"Unique AircraftAddress: {uniq_aircraft:,}")

    print_section("STATUS DISTRIBUTION (top 20)")
    rows = fetch_all(
        con,
        f"""
        SELECT Status, COUNT(*) AS c
        FROM AsterixPackets
        WHERE 1=1 {FILTER_SQL}
        GROUP BY Status
        ORDER BY c DESC
        LIMIT 20
        """,
    )
    for status, c in rows:
        print(f"Status={status}: {c:,}")

    print_section("TOP AIRCRAFT BY PACKET COUNT (top 20)")
    rows = fetch_all(
        con,
        f"""
        SELECT AircraftAddress, COUNT(*) AS c
        FROM AsterixPackets
        WHERE AircraftAddress IS NOT NULL {FILTER_SQL}
        GROUP BY AircraftAddress
        ORDER BY c DESC
        LIMIT 20
        """,
    )
    for addr, c in rows:
        print(f"AircraftAddress={addr}: {c:,}")

    print_section("NUMERIC SUMMARY (SQL aggregates)")
    agg = fetch_one(
        con,
        f"""
        SELECT
          MIN(VelocitySpeed), AVG(VelocitySpeed), MAX(VelocitySpeed),
          MIN(VelocityHeading), AVG(VelocityHeading), MAX(VelocityHeading),
          MIN(FlightLevelFeet), AVG(FlightLevelFeet), MAX(FlightLevelFeet)
        FROM AsterixPackets
        WHERE 1=1 {FILTER_SQL}
        """,
    )
    (vs_min, vs_avg, vs_max, hdg_min, hdg_avg, hdg_max, fl_min, fl_avg, fl_max) = agg
    print(f"VelocitySpeed    min/avg/max: {vs_min} / {vs_avg:.3f} / {vs_max}")
    print(f"VelocityHeading  min/avg/max: {hdg_min} / {hdg_avg:.3f} / {hdg_max}")
    print(f"FlightLevelFeet  min/avg/max: {fl_min} / {fl_avg:.3f} / {fl_max}")


def print_percentiles_from_sample(con: sqlite3.Connection):
    print_section("APPROX PERCENTILES (sample via Id%step==0)")

    step = approx_sample_step(con, SAMPLE_TARGET)
    print(f"Sampling step: Id % {step} == 0 (approx target {SAMPLE_TARGET:,})")

    # Sample numeric columns
    sql = f"""
        SELECT VelocitySpeed, FlightLevelFeet
        FROM AsterixPackets
        WHERE (Id % ?) = 0 {FILTER_SQL}
          AND VelocitySpeed IS NOT NULL
          AND FlightLevelFeet IS NOT NULL
    """
    rows = con.execute(sql, (step,)).fetchall()
    if not rows:
        print("No sample rows returned; check FILTER_SQL or table contents.")
        return

    data = np.asarray(rows, dtype=np.float64)
    vs = data[:, 0]
    fl = data[:, 1]

    for name, arr in [("VelocitySpeed", vs), ("FlightLevelFeet", fl)]:
        p = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name} percentiles [1,5,25,50,75,95,99]: {np.array2string(p, precision=3)}")


# =========================
# PLOTS
# =========================
def plot_position_density_and_mean_alt(con: sqlite3.Connection, out_dir: str):
    print_section("PLOT: POSITION DENSITY HEATMAP + MEAN ALTITUDE MAP")

    # Pass 1: bounds
    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    any_rows = False
    for chunk in iter_rows_positions_alt(con, CHUNK_ROWS):
        any_rows = True
        rho = chunk[:, 0]
        theta = chunk[:, 1]
        x, y = polar_to_xy_azimuth(rho, theta)

        xmin = min(xmin, float(np.min(x)))
        xmax = max(xmax, float(np.max(x)))
        ymin = min(ymin, float(np.min(y)))
        ymax = max(ymax, float(np.max(y)))

    if not any_rows:
        print("No rows for position/altitude; skipping heatmaps.")
        return

    x_edges = np.linspace(xmin, xmax, BINS_POS + 1)
    y_edges = np.linspace(ymin, ymax, BINS_POS + 1)

    # Accumulators
    density = np.zeros((BINS_POS, BINS_POS), dtype=np.uint64)
    sum_alt = np.zeros((BINS_POS, BINS_POS), dtype=np.float64)
    cnt_alt = np.zeros((BINS_POS, BINS_POS), dtype=np.uint32)

    for chunk in iter_rows_positions_alt(con, CHUNK_ROWS):
        rho = chunk[:, 0]
        theta = chunk[:, 1]
        alt_ft = chunk[:, 2]

        x, y = polar_to_xy_azimuth(rho, theta)

        # density via histogram2d
        h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        density += h.astype(np.uint64)

        # mean altitude via bin indices + add.at
        ix = np.searchsorted(x_edges, x, side="right") - 1
        iy = np.searchsorted(y_edges, y, side="right") - 1
        valid = (ix >= 0) & (ix < BINS_POS) & (iy >= 0) & (iy < BINS_POS) & np.isfinite(alt_ft)
        ix = ix[valid].astype(np.intp)
        iy = iy[valid].astype(np.intp)
        alt_ft = alt_ft[valid]

        np.add.at(sum_alt, (ix, iy), alt_ft)
        np.add.at(cnt_alt, (ix, iy), 1)

    # Density plot (log)
    dens_log = np.log1p(density)

    plt.figure(figsize=(10, 8))
    plt.imshow(
        dens_log.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
    )
    plt.colorbar(label="log(1 + count)")
    plt.title("Position Density Heatmap")
    plt.xlabel("x (East)")
    plt.ylabel("y (North)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pos_density_heatmap.png"), dpi=160)
    plt.close()

    # Mean altitude plot
    mean_alt = np.full_like(sum_alt, np.nan, dtype=np.float64)
    mask = cnt_alt >= MIN_COUNT_PER_BIN_MEAN_ALT
    mean_alt[mask] = sum_alt[mask] / cnt_alt[mask]

    plt.figure(figsize=(10, 8))
    plt.imshow(
        mean_alt.T,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
    )
    plt.colorbar(label="Mean FlightLevelFeet (ft)")
    plt.title(f"Mean Altitude per Position (min {MIN_COUNT_PER_BIN_MEAN_ALT} samples/bin)")
    plt.xlabel("x (East)")
    plt.ylabel("y (North)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pos_mean_altitude.png"), dpi=160)
    plt.close()

    print(f"Saved: {os.path.join(out_dir, 'pos_density_heatmap.png')}")
    print(f"Saved: {os.path.join(out_dir, 'pos_mean_altitude.png')}")


def plot_time_deltas(con: sqlite3.Connection, out_dir: str):
    print_section("PLOT: TIME DELTA HISTOGRAM (per AircraftAddress, seconds)")

    # Δt per AircraftAddress. If you prefer global: remove PARTITION BY.
    query = f"""
    WITH t AS (
      SELECT
        Id,
        AircraftAddress,
        (julianday(TimeOfDay) - julianday('1980-01-01 00:00:00+00:00')) * 86400.0 AS t_s
      FROM AsterixPackets
      WHERE TimeOfDay IS NOT NULL
        AND AircraftAddress IS NOT NULL
        {FILTER_SQL}
    ),
    d AS (
      SELECT
        (t_s - LAG(t_s) OVER (
          PARTITION BY AircraftAddress
          ORDER BY Id
        )) AS dt_s
      FROM t
    )
    SELECT dt_s
    FROM d
    WHERE dt_s IS NOT NULL;
    """

    counts = np.zeros(len(DT_BINS_S) - 1, dtype=np.int64)
    nonpos = 0
    total_pos = 0

    cur = con.cursor()
    cur.execute(query)

    while True:
        rows = cur.fetchmany(CHUNK_ROWS)
        if not rows:
            break

        dt_s = np.fromiter((r[0] for r in rows), dtype=np.float64, count=len(rows))
        dt_s = dt_s[np.isfinite(dt_s)]

        nonpos += int(np.sum(dt_s <= 0))
        dt_s = dt_s[dt_s > 0]
        if dt_s.size == 0:
            continue

        total_pos += dt_s.size
        h, _ = np.histogram(dt_s, bins=DT_BINS_S)
        counts += h

    x = np.sqrt(DT_BINS_S[:-1] * DT_BINS_S[1:])

    plt.figure(figsize=(10, 6))
    plt.plot(x, counts)
    plt.xscale("log")
    if counts.max() > 0:
        plt.yscale("log")
    plt.xlabel("Δt (seconds)")
    plt.ylabel("Count")
    plt.title(f"Δt Histogram per AircraftAddress (dt>0: {total_pos:,}, dt<=0: {nonpos:,})")
    plt.tight_layout()
    path = os.path.join(out_dir, "time_delta_hist_seconds.png")
    plt.savefig(path, dpi=160)
    plt.close()

    print(f"Saved: {path}")


def plot_speed_heading_flightlevel(con: sqlite3.Connection, out_dir: str):
    print_section("PLOT: SPEED / HEADING / FLIGHTLEVEL DISTRIBUTIONS")

    # Accumulate histograms chunkwise (avoid holding 14M values)
    # Choose reasonable ranges; you can adjust if your data differs.
    speed_bins = np.linspace(0, 400, 200)      # units depend on your data
    fl_bins = np.linspace(0, 50000, 200)       # feet
    hdg_bins = np.linspace(0, 360, 361)        # degrees 0..360

    speed_counts = np.zeros(len(speed_bins) - 1, dtype=np.int64)
    fl_counts = np.zeros(len(fl_bins) - 1, dtype=np.int64)
    hdg_counts = np.zeros(len(hdg_bins) - 1, dtype=np.int64)

    for chunk in iter_rows_speeds_headings_fls(con, CHUNK_ROWS):
        speed = chunk[:, 0]
        heading = chunk[:, 1]
        fl = chunk[:, 2]

        speed = speed[np.isfinite(speed)]
        heading = heading[np.isfinite(heading)]
        fl = fl[np.isfinite(fl)]

        # Normalize heading into [0, 360)
        if heading.size:
            heading = np.mod(heading, 360.0)

        speed_counts += np.histogram(speed, bins=speed_bins)[0]
        fl_counts += np.histogram(fl, bins=fl_bins)[0]
        hdg_counts += np.histogram(heading, bins=hdg_bins)[0]

    # Speed histogram
    xc = 0.5 * (speed_bins[:-1] + speed_bins[1:])
    plt.figure(figsize=(10, 6))
    plt.plot(xc, speed_counts)
    plt.yscale("log") if speed_counts.max() > 0 else None
    plt.xlabel("VelocitySpeed")
    plt.ylabel("Count")
    plt.title("VelocitySpeed Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, "velocity_speed_hist.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved: {path}")

    # Flight level histogram
    xc = 0.5 * (fl_bins[:-1] + fl_bins[1:])
    plt.figure(figsize=(10, 6))
    plt.plot(xc, fl_counts)
    plt.yscale("log") if fl_counts.max() > 0 else None
    plt.xlabel("FlightLevelFeet (ft)")
    plt.ylabel("Count")
    plt.title("FlightLevelFeet Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, "flightlevelfeet_hist.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved: {path}")

    # Heading polar "rose"
    # Use bin centers in radians, weights are counts
    centers_deg = 0.5 * (hdg_bins[:-1] + hdg_bins[1:])
    centers_rad = np.deg2rad(centers_deg)
    widths = np.deg2rad(np.diff(hdg_bins))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(centers_rad, hdg_counts, width=widths, align="center")
    ax.set_theta_zero_location("N")  # 0 at north
    ax.set_theta_direction(-1)       # clockwise
    ax.set_title("Heading Distribution (Polar)")
    fig.tight_layout()
    path = os.path.join(out_dir, "heading_polar.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


# =========================
# MAIN
# =========================
def main():
    ensure_out_dir(OUT_DIR)
    con = connect(DB_PATH)

    try:
        print_overview(con)
        print_percentiles_from_sample(con)

        plot_position_density_and_mean_alt(con, OUT_DIR)
        plot_time_deltas(con, OUT_DIR)
        plot_speed_heading_flightlevel(con, OUT_DIR)

        print_section("DONE")
        print(f"All outputs written to: {os.path.abspath(OUT_DIR)}")

        print("\nOptional index (recommended for per-aircraft dt queries):")
        print("  CREATE INDEX IF NOT EXISTS IX_AsterixPackets_AircraftAddress_Id ON AsterixPackets(AircraftAddress, Id);")

    finally:
        con.close()


if __name__ == "__main__":
    main()
