import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "C:\\Projekte\\radar-simulation-validator\\RadarSimulationValidator\\backup-31.12.2025.db"


CHUNK = 500_000
WHERE = "WHERE Status = 1"  # e.g. 'WHERE Status = 1'

# Plot tuning
POINT_SIZE = 1     # smaller = faster/cleaner for huge point clouds
ALPHA = 1         # transparency helps show density
DOWNSAMPLE = None    # e.g. set to 200_000 to cap points per chunk, or None for all

def iter_rows(conn, chunk=CHUNK):
    cur = conn.cursor()
    cur.execute(f"SELECT MIN(Id), MAX(Id) FROM AsterixPackets {WHERE}")
    min_id, max_id = cur.fetchone()

    if min_id is None:
        return

    start = min_id
    while start <= max_id:
        end = start + chunk - 1
        cur.execute(
            f"""
            SELECT PositionRho, PositionTheta
            FROM AsterixPackets
            {WHERE}
            AND Id BETWEEN ? AND ?
            """ if WHERE else
            """
            SELECT PositionRho, PositionTheta
            FROM AsterixPackets
            WHERE Id BETWEEN ? AND ?
            """,
            (start, end)
        )
        rows = cur.fetchall()
        print(f"Loaded rows {start} to {end}, got {len(rows)} rows")
        if rows:
            yield np.array(rows, dtype=np.float64)
        start = end + 1

conn = sqlite3.connect(DB_PATH)

plt.figure(figsize=(10, 8))

total_plotted = 0

for data in iter_rows(conn):
    rho = data[:, 0]
    theta = np.deg2rad(data[:, 1])

    x = rho * np.sin(theta)
    y = rho * np.cos(theta)

    # Optional downsampling per chunk to keep plotting responsive
    if DOWNSAMPLE is not None and x.size > DOWNSAMPLE:
        idx = np.random.choice(x.size, size=DOWNSAMPLE, replace=False)
        x = x[idx]
        y = y[idx]

    plt.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, linewidths=0)
    total_plotted += x.size

conn.close()

plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"All Positions (scatter) — plotted {total_plotted:,} points")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
