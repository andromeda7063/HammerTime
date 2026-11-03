import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from physical_graph import build_physical_circuit_edges


@njit(cache=False, fastmath=False)
def edge_numeric(d, speed_kmh, active, indices, lap, max_gap, sigma, coverage):
    n = len(d)
    src = np.empty(n, np.int64)
    dst = np.empty(n, np.int64)
    attrs = np.empty((n, 5), np.float64)
    count = blocked = 0
    for i in range(n):
        j = (i + 1) % n
        gap = (d[j] - d[i]) % lap
        if not math.isfinite(gap) or gap <= 0.0:
            continue
        vf = speed_kmh[i] / 3.6
        vl = speed_kmh[j] / 3.6
        floor = vf if vf >= 10.0 else 10.0
        seconds = gap / floor
        if not (0.0 < seconds <= max_gap):
            continue
        if not active[i] or not active[j]:
            blocked += 1
            continue
        src[count], dst[count] = indices[j], indices[i]
        attrs[count, 0] = math.exp(-(seconds ** 2) / (2.0 * sigma ** 2))
        attrs[count, 1] = gap
        attrs[count, 2] = vl - vf
        attrs[count, 3] = coverage
        attrs[count, 4] = 1.0 if (gap <= 35.0 and seconds <= 0.6) else 0.0
        count += 1
    return src[:count], dst[:count], attrs[:count], blocked


def candidate(args):
    distance, speed, valid, active, driver_to_idx, lap, max_gap, sigma, expected = args
    eligible = [
        str(driver) for driver in driver_to_idx
        if valid.get(driver, False)
        and driver in distance and driver in speed
        and math.isfinite(float(distance[driver]))
        and math.isfinite(float(speed[driver]))
    ]
    expected_count = len(driver_to_idx) if expected is None else max(0, int(expected))
    coverage = float(len(eligible) / expected_count) if expected_count else 0.0
    ordered = sorted(eligible, key=lambda name: (float(distance[name]) % float(lap), name))
    if len(ordered) <= 1:
        edge = build_physical_circuit_edges(*args, edge_schema="physical_v2")
        return edge.edge_index, edge.edge_attr, edge.position_coverage, edge.blocked_neighbor_count
    d = np.asarray([float(distance[name]) % float(lap) for name in ordered], dtype=np.float64)
    s = np.asarray([float(speed[name]) for name in ordered], dtype=np.float64)
    a = np.asarray([bool(active.get(name, False)) for name in ordered], dtype=np.bool_)
    ix = np.asarray([int(driver_to_idx[name]) for name in ordered], dtype=np.int64)
    src, dst, attrs, blocked = edge_numeric(
        d, s, a, ix, float(lap), float(max_gap), float(sigma), coverage
    )
    edge_index = torch.as_tensor(np.stack((src, dst)), dtype=torch.long) if len(src) else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.as_tensor(attrs, dtype=torch.float32) if len(src) else torch.empty((0, 5), dtype=torch.float32)
    return edge_index, edge_attr, coverage, blocked


def make_frames(count=1000, packed=True):
    rng = np.random.default_rng(431)
    frames = []
    ids = [f"D{i:02d}" for i in range(20)]
    lap = 5200.0
    for frame in range(count):
        if packed:
            positions = (np.arange(20) * 40.0 + rng.normal(0, 1, 20) + frame * 3.0) % lap
        else:
            positions = (np.arange(20) * lap / 20 + rng.normal(0, 30, 20) + frame * 3.0) % lap
        distance = dict(zip(ids, positions.tolist()))
        speed = dict(zip(ids, (180 + rng.normal(0, 25, 20)).tolist()))
        valid = {name: True for name in ids}
        active = {name: int(name[1:]) % 7 != 0 for name in ids}
        driver_to_idx = {name: i for i, name in enumerate(ids)}
        frames.append((distance, speed, valid, active, driver_to_idx, lap, 2.0, 1.0, 20))
    # Deterministic contract cases: wrap adjacency, equal positions, exact
    # gap-time boundary, physical-v2 mask, missing/inactive data, and NaN.
    for case in range(6):
        distance = {name: float((5200.0 - 40.0 + i * 40.0) % 5200.0) for i, name in enumerate(ids)}
        speed = {name: 100.0 for name in ids}
        valid = {name: True for name in ids}
        active = {name: True for name in ids}
        if case == 0:  # put one adjacent pair across lap zero
            distance[ids[0]], distance[ids[1]] = 5190.0, 10.0
        elif case == 1:  # equal-distance tie must keep driver lexical ordering
            distance[ids[0]] = distance[ids[1]]
        elif case == 2:  # exercise the <= 2 s gap-time boundary
            distance[ids[0]], distance[ids[1]] = 100.0, 100.0 + (100.0 / 3.6) * 2.0
        elif case == 3:  # exercise the <= 35 m / <= 0.6 s physical-v2 mask
            distance[ids[0]], distance[ids[1]] = 100.0, 135.0
            speed[ids[0]] = speed[ids[1]] = 210.0
        elif case == 4:  # missing/inactive fields plus zero-speed floor
            valid[ids[3]] = False
            active[ids[4]] = False
            speed[ids[5]] = 0.0
            distance.pop(ids[6])
            speed.pop(ids[7])
        else:  # a valid flag must not admit a nonfinite speed into the kernel
            speed[ids[8]] = math.nan
        driver_to_idx = {name: i for i, name in enumerate(ids)}
        frames.append((distance, speed, valid, active, driver_to_idx, lap, 2.0, 1.0, 20))
    return frames


def baseline(args):
    return build_physical_circuit_edges(*args, edge_schema="physical_v2")


def run(packed, frames_count, repeats):
    frames = make_frames(count=frames_count, packed=packed)
    t0 = time.perf_counter()
    first_candidate = candidate(frames[0])
    cold = time.perf_counter() - t0
    first_ref = baseline(frames[0])
    assert torch.equal(first_candidate[0], first_ref.edge_index)
    assert torch.equal(first_candidate[1], first_ref.edge_attr)
    assert first_candidate[2:] == (first_ref.position_coverage, first_ref.blocked_neighbor_count)
    total_edges = int(first_ref.edge_index.shape[1])
    for f in frames[1:]:
        assert sum(
            bool(f[2].get(name, False)) and name in f[0] and name in f[1]
            and math.isfinite(float(f[0][name])) and math.isfinite(float(f[1][name]))
            for name in f[4]
        ) > 0
        ref, got = baseline(f), candidate(f)
        assert torch.equal(got[0], ref.edge_index)
        assert torch.equal(got[1], ref.edge_attr)
        assert got[2:] == (ref.position_coverage, ref.blocked_neighbor_count)
        total_edges += int(ref.edge_index.shape[1])
    def bench(fn):
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            for f in frames:
                fn(f)
            values.append(time.perf_counter() - start)
        return values
    reference = bench(baseline)
    compiled = bench(candidate)
    reference_summary = {
        "median_seconds": float(np.median(reference)),
        "min_seconds": float(np.min(reference)),
        "max_seconds": float(np.max(reference)),
    }
    compiled_summary = {
        "median_seconds": float(np.median(compiled)),
        "min_seconds": float(np.min(compiled)),
        "max_seconds": float(np.max(compiled)),
    }
    median_ratio = compiled_summary["median_seconds"] / reference_summary["median_seconds"]
    return {
        "fixture_only": True,
        "scenario": "packed" if packed else "evenly_spaced",
        "frames": len(frames),
        "cars_per_frame": 20,
        "edges_first_frame": int(first_ref.edge_index.shape[1]),
        "edges_total": total_edges,
        "first_candidate_call_seconds": cold,
        "baseline_seconds": reference_summary,
        "candidate_warm_seconds": compiled_summary,
        "candidate_to_baseline_median_ratio": median_ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic physical-graph Numba microbenchmark; not race data.")
    parser.add_argument("--frames", type=int, default=1000, help="generated frames per density case, before six contract cases")
    parser.add_argument("--repeats", type=int, default=50, help="timed passes over each case after correctness checks")
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error("--frames and --repeats must be positive")
    print(run(True, args.frames, args.repeats))
    print(run(False, args.frames, args.repeats))
