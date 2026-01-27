#!/usr/bin/env python3
"""
rest_bench.py - Vergleich mehrerer REST-Endpunkte anhand Performance-Kennzahlen.

Beispiele:
  python rest_bench.py --endpoints endpoints.json --n 200 --c 20 --timeout 5
  python rest_bench.py --endpoints endpoints.json --n 500 --c 50 --warmup 50 --method GET
  python rest_bench.py --endpoints endpoints.json --n 200 --c 20 --out-json results.json --out-csv results.csv

endpoints.json Beispiel:
[
  {"name":"Service A /health", "url":"https://service-a.example.com/health"},
  {"name":"Service B /health", "url":"https://service-b.example.com/health"}
]

Optional je Endpoint:
  - method, headers, params, body (string) oder body_file, expected_status (int oder Liste),
    verify (bool), follow_redirects (bool)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics as stats
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx


@dataclass
class EndpointSpec:
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = None
    params: Dict[str, str] = None
    body: Optional[str] = None
    body_file: Optional[str] = None
    expected_status: Optional[Union[int, List[int]]] = None
    verify: bool = True
    follow_redirects: bool = True

    def normalized(self) -> "EndpointSpec":
        return EndpointSpec(
            name=self.name,
            url=self.url,
            method=(self.method or "GET").upper(),
            headers=self.headers or {},
            params=self.params or {},
            body=self.body,
            body_file=self.body_file,
            expected_status=self.expected_status,
            verify=self.verify if self.verify is not None else True,
            follow_redirects=self.follow_redirects if self.follow_redirects is not None else True,
        )

    def load_body(self) -> Optional[bytes]:
        if self.body_file:
            with open(self.body_file, "rb") as f:
                return f.read()
        if self.body is not None:
            return self.body.encode("utf-8")
        return None


@dataclass
class Sample:
    ok: bool
    status_code: Optional[int]
    latency_ms: float
    bytes_in: int
    error: Optional[str] = None


@dataclass
class Summary:
    name: str
    url: str
    method: str
    total_requests: int
    concurrency: int
    timeout_s: float
    warmup: int
    duration_s: float
    success_rate: float
    ok_count: int
    fail_count: int
    timeout_count: int
    error_count: int
    status_counts: Dict[str, int]
    bytes_in_total: int
    rps: float
    latency_mean_ms: float
    latency_stdev_ms: float
    latency_min_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile with linear interpolation."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def parse_endpoints(path: str) -> List[EndpointSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("endpoints Datei muss eine JSON-Liste sein.")
    specs: List[EndpointSpec] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Endpoint #{i} ist kein Objekt.")
        if "name" not in item or "url" not in item:
            raise ValueError(f"Endpoint #{i} braucht mindestens 'name' und 'url'.")
        specs.append(EndpointSpec(**item).normalized())
    return specs


def is_expected(status_code: int, expected: Optional[Union[int, List[int]]]) -> bool:
    if expected is None:
        return 200 <= status_code <= 299
    if isinstance(expected, int):
        return status_code == expected
    return status_code in expected


async def one_request(
    client: httpx.AsyncClient,
    spec: EndpointSpec,
    timeout_s: float,
) -> Sample:
    body = spec.load_body()
    t0 = time.perf_counter()
    try:
        resp = await client.request(
            spec.method,
            spec.url,
            headers=spec.headers,
            params=spec.params,
            content=body,
            timeout=timeout_s,
        )
        t1 = time.perf_counter()
        ok = is_expected(resp.status_code, spec.expected_status)
        # bytes_in: Content-Length ist oft gesetzt, ansonsten len(content)
        content = resp.content  # lädt Body (wichtig, sonst messen wir unrealistisch)
        bytes_in = len(content) if content is not None else 0
        return Sample(
            ok=ok,
            status_code=resp.status_code,
            latency_ms=(t1 - t0) * 1000.0,
            bytes_in=bytes_in,
            error=None if ok else f"unexpected_status:{resp.status_code}",
        )
    except httpx.ReadTimeout:
        t1 = time.perf_counter()
        return Sample(ok=False, status_code=None, latency_ms=(t1 - t0) * 1000.0, bytes_in=0, error="timeout")
    except httpx.ConnectTimeout:
        t1 = time.perf_counter()
        return Sample(ok=False, status_code=None, latency_ms=(t1 - t0) * 1000.0, bytes_in=0, error="connect_timeout")
    except httpx.ConnectError as e:
        t1 = time.perf_counter()
        return Sample(ok=False, status_code=None, latency_ms=(t1 - t0) * 1000.0, bytes_in=0, error=f"connect_error:{type(e).__name__}")
    except httpx.RemoteProtocolError as e:
        t1 = time.perf_counter()
        return Sample(ok=False, status_code=None, latency_ms=(t1 - t0) * 1000.0, bytes_in=0, error=f"protocol_error:{type(e).__name__}")
    except Exception as e:
        t1 = time.perf_counter()
        return Sample(ok=False, status_code=None, latency_ms=(t1 - t0) * 1000.0, bytes_in=0, error=f"error:{type(e).__name__}")


async def run_benchmark_for_endpoint(
    spec: EndpointSpec,
    n: int,
    concurrency: int,
    timeout_s: float,
    warmup: int,
) -> Tuple[Summary, List[Sample]]:
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    async with httpx.AsyncClient(
        verify=spec.verify,
        follow_redirects=spec.follow_redirects,
        limits=limits,
        headers=spec.headers,
    ) as client:

        # Warmup (nicht in Samples)
        if warmup > 0:
            warm_tasks = [one_request(client, spec, timeout_s) for _ in range(warmup)]
            # begrenze Parallelität auch im Warmup
            for i in range(0, len(warm_tasks), concurrency):
                await asyncio.gather(*warm_tasks[i : i + concurrency])

        samples: List[Sample] = []
        sem = asyncio.Semaphore(concurrency)

        async def worker() -> None:
            async with sem:
                s = await one_request(client, spec, timeout_s)
                samples.append(s)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(n)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    duration_s = max(1e-9, (t_end - t_start))
    ok_samples = [s for s in samples if s.ok]
    lat_ok = [s.latency_ms for s in ok_samples]

    status_counts: Dict[str, int] = {}
    timeout_count = 0
    error_count = 0
    for s in samples:
        if s.error == "timeout" or s.error == "connect_timeout":
            timeout_count += 1
        if s.error and not s.error.startswith("unexpected_status"):
            error_count += 1
        if s.status_code is None:
            key = "NO_STATUS"
        else:
            key = str(s.status_code)
        status_counts[key] = status_counts.get(key, 0) + 1

    ok_count = len(ok_samples)
    fail_count = n - ok_count
    bytes_in_total = sum(s.bytes_in for s in samples)

    # Falls alle Requests failen, sind Perzentile/Mean NaN (besser als 0, weil sonst missverständlich)
    latency_mean = stats.mean(lat_ok) if lat_ok else float("nan")
    latency_stdev = stats.pstdev(lat_ok) if len(lat_ok) >= 2 else (0.0 if len(lat_ok) == 1 else float("nan"))
    latency_min = min(lat_ok) if lat_ok else float("nan")
    latency_max = max(lat_ok) if lat_ok else float("nan")

    summary = Summary(
        name=spec.name,
        url=spec.url,
        method=spec.method,
        total_requests=n,
        concurrency=concurrency,
        timeout_s=timeout_s,
        warmup=warmup,
        duration_s=duration_s,
        success_rate=(ok_count / n) if n > 0 else 0.0,
        ok_count=ok_count,
        fail_count=fail_count,
        timeout_count=timeout_count,
        error_count=error_count,
        status_counts=status_counts,
        bytes_in_total=bytes_in_total,
        rps=n / duration_s,
        latency_mean_ms=latency_mean,
        latency_stdev_ms=latency_stdev,
        latency_min_ms=latency_min,
        latency_p50_ms=percentile(lat_ok, 50),
        latency_p90_ms=percentile(lat_ok, 90),
        latency_p95_ms=percentile(lat_ok, 95),
        latency_p99_ms=percentile(lat_ok, 99),
        latency_max_ms=latency_max,
    )
    return summary, samples


def fmt_ms(x: float) -> str:
    if x != x:  # NaN check
        return "NaN"
    return f"{x:8.2f}"


def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def print_summary_table(summaries: List[Summary]) -> None:
    # Sortierung: primär Success-Rate desc, dann p95 asc
    summaries_sorted = sorted(
        summaries,
        key=lambda s: (-s.success_rate, s.latency_p95_ms if s.latency_p95_ms == s.latency_p95_ms else float("inf")),
    )

    print("\n=== Benchmark Summary (sorted) ===")
    header = (
        "Name",
        "OK%",
        "RPS",
        "mean",
        "p50",
        "p95",
        "p99",
        "min",
        "max",
        "fails",
        "timeouts",
        "bytes_in",
    )
    print(
        f"{header[0]:30} {header[1]:>7} {header[2]:>8} {header[3]:>9} {header[4]:>9} {header[5]:>9} "
        f"{header[6]:>9} {header[7]:>9} {header[8]:>9} {header[9]:>7} {header[10]:>9} {header[11]:>10}"
    )
    print("-" * 130)
    for s in summaries_sorted:
        print(
            f"{s.name[:30]:30} {fmt_pct(s.success_rate):>7} {s.rps:8.2f} {fmt_ms(s.latency_mean_ms):>9} "
            f"{fmt_ms(s.latency_p50_ms):>9} {fmt_ms(s.latency_p95_ms):>9} {fmt_ms(s.latency_p99_ms):>9} "
            f"{fmt_ms(s.latency_min_ms):>9} {fmt_ms(s.latency_max_ms):>9} {s.fail_count:7d} "
            f"{s.timeout_count:9d} {s.bytes_in_total:10d}"
        )

    print("\nHinweise:")
    print("- Latenzen beziehen sich auf erfolgreiche Requests (OK gemäß expected_status oder 2xx Default).")
    print("- RPS = total_requests / gemessene Dauer (Warmup nicht eingerechnet).")
    print("- bytes_in ist Summe Response-Body (kann bei großen Responses dominieren).")


def print_status_breakdown(s: Summary) -> None:
    items = sorted(s.status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = items[:12]
    rest = items[12:]
    print(f"\nStatuscodes für '{s.name}':")
    for k, v in top:
        print(f"  {k:10} {v}")
    if rest:
        print(f"  ... ({len(rest)} weitere)")


def write_json(path: str, summaries: List[Summary]) -> None:
    payload = [asdict(s) for s in summaries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: str, summaries: List[Summary]) -> None:
    fields = [
        "name", "url", "method",
        "total_requests", "concurrency", "timeout_s", "warmup",
        "duration_s", "success_rate", "ok_count", "fail_count", "timeout_count", "error_count",
        "rps", "bytes_in_total",
        "latency_mean_ms", "latency_stdev_ms", "latency_min_ms", "latency_p50_ms",
        "latency_p90_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            row = {k: getattr(s, k) for k in fields}
            w.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark für mehrere REST-Endpunkte (Latenz/Throughput/Fehler).")
    p.add_argument("--endpoints", required=True, help="Pfad zu JSON-Datei mit Endpoints (Liste).")
    p.add_argument("--n", type=int, default=200, help="Anzahl Requests pro Endpoint (Default: 200).")
    p.add_argument("--c", type=int, default=20, help="Concurrency pro Endpoint (Default: 20).")
    p.add_argument("--timeout", type=float, default=5.0, help="Timeout in Sekunden pro Request (Default: 5).")
    p.add_argument("--warmup", type=int, default=20, help="Warmup-Requests pro Endpoint (Default: 20).")
    p.add_argument("--show-status", action="store_true", help="Statuscode-Verteilung pro Endpoint ausgeben.")
    p.add_argument("--out-json", default=None, help="Optional: Ergebnisse als JSON speichern.")
    p.add_argument("--out-csv", default=None, help="Optional: Ergebnisse als CSV speichern.")
    return p


async def main_async(args: argparse.Namespace) -> int:
    specs = parse_endpoints(args.endpoints)

    summaries: List[Summary] = []
    for spec in specs:
        print(f"\n--- Running: {spec.name} ({spec.method} {spec.url}) ---")
        summary, _samples = await run_benchmark_for_endpoint(
            spec=spec,
            n=args.n,
            concurrency=args.c,
            timeout_s=args.timeout,
            warmup=args.warmup,
        )
        summaries.append(summary)

        print(f"Done: OK={fmt_pct(summary.success_rate)}  RPS={summary.rps:.2f}  p95={summary.latency_p95_ms:.2f}ms  duration={summary.duration_s:.2f}s")
        if args.show_status:
            print_status_breakdown(summary)

    print_summary_table(summaries)

    if args.out_json:
        write_json(args.out_json, summaries)
        print(f"\nWrote JSON: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, summaries)
        print(f"Wrote CSV: {args.out_csv}")

    return 0


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.n <= 0 or args.c <= 0:
        raise SystemExit("--n und --c müssen > 0 sein.")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
