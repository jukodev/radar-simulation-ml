import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = "C:\\Users\\theju\\Downloads\\backup-17.12.2025.db"

CHUNK = 500_000

query = """
WITH t AS (
  SELECT
    Id,
    FlightId,
    (julianday(TimeOfDay) - julianday('1980-01-01 00:00:00+00:00')) * 86400.0 AS t_s
  FROM AsterixPackets
  WHERE TimeOfDay IS NOT NULL AND Status = 1
),
d AS (
  SELECT
    (t_s - LAG(t_s) OVER (
      PARTITION BY FlightId
      ORDER BY Id
    )) AS dt_s
  FROM t
)
SELECT dt_s
FROM d
WHERE dt_s IS NOT NULL;
"""

# Bins in seconds: 1s .. 2h (anpassen)
bins_s = np.logspace(0, np.log10(7200), 220)  # 10^0 = 1s
counts = np.zeros(len(bins_s) - 1, dtype=np.int64)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()
cur.execute(query)

nonpos = 0
total = 0

while True:
    rows = cur.fetchmany(CHUNK)
    if not rows:
        break

    dt_s = np.fromiter((r[0] for r in rows), dtype=np.float64, count=len(rows))
    dt_s = dt_s[np.isfinite(dt_s)]

    nonpos += int(np.sum(dt_s <= 0))
    dt_s = dt_s[dt_s > 0]  # required for log bins

    if dt_s.size == 0:
        continue

    total += dt_s.size
    h, _ = np.histogram(dt_s, bins=bins_s)
    counts += h

con.close()

x = np.sqrt(bins_s[:-1] * bins_s[1:])

plt.figure(figsize=(10, 6))
plt.plot(x, counts)
plt.xscale("log")
if counts.max() > 0:
    plt.yscale("log")
plt.xlabel("Δt (Sekunden)")
plt.ylabel("Anzahl")
plt.title(f"Δt Histogramm (dt>0: {total:,}, dt<=0: {nonpos:,})")
plt.tight_layout()
plt.show()
