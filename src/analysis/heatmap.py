import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "C:\\Users\\theju\\Downloads\\backup-17.12.2025.db"

# Raster-Auflösung (je höher, desto feiner, desto mehr RAM/CPU)
BINS = 800  # z.B. 400..2000 je nach Daten/Performance

CHUNK = 500_000

WHERE = ""  # z.B. 'WHERE Status = 1'

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

# First pass: Determine bounds
xmin = ymin = np.inf
xmax = ymax = -np.inf

for data in iter_rows(conn):
    rho = data[:, 0]
    theta = data[:, 1]

    theta = np.deg2rad(theta)

    x = rho * np.sin(theta)
    y =  rho * np.cos(theta)

    xmin = min(xmin, x.min())
    xmax = max(xmax, x.max())
    ymin = min(ymin, y.min())
    ymax = max(ymax, y.max())

# Second pass: Accumulate histogram
H = np.zeros((BINS, BINS), dtype=np.uint64)

x_edges = np.linspace(xmin, xmax, BINS + 1)
y_edges = np.linspace(ymin, ymax, BINS + 1)

for data in iter_rows(conn):
    rho = data[:, 0]
    theta = data[:, 1]
    theta = np.deg2rad(theta)

    x = rho * np.sin(theta)
    y = rho * np.cos(theta)

    h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    H += h.astype(np.uint64)

conn.close()

# Plot heatmap
H_log = np.log1p(H)  # log(1 + count)

plt.figure(figsize=(10, 8))
plt.imshow(
    H_log.T,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",
)
plt.colorbar(label="log(1 + count)")
plt.title("Position Heatmap (binned density)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
