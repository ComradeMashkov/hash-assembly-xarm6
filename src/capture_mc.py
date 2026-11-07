"""Capture experiment: how much robot placement error does the passive fixture absorb?

A part is released above its seat with a random pose error, then the jaws close.
Results describe capture geometry in an idealised rigid-contact model, not micron accuracy.
"""
import csv
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from cell import PARTS
from run_cell import CTRL_DT, DROP, SIDE_CLAMP_DELAY, Cell

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRIALS = 600
MAX_ERROR_MM = 16
MAX_YAW = np.radians(1.0)
BY_NAME = {p.name: p for p in PARTS}


def set_part(cell, p, dx=0.0, dy=0.0, dyaw=0.0, lift=0.0):
    adr = cell.m.joint(f"{p.name}_free").qposadr[0]
    half = (p.yaw + dyaw) / 2
    cell.d.qpos[adr:adr + 7] = [p.x + dx, p.y + dy, p.z + p.height / 2 + lift,
                                np.cos(half), 0, 0, np.sin(half)]


def settle(cell, seconds):
    mujoco.mj_step(cell.m, cell.d, nstep=int(seconds / cell.m.opt.timestep))


def clamp(cell, p):
    cell.close_jaws(p, "e")
    settle(cell, SIDE_CLAMP_DELAY)
    cell.close_jaws(p, "s")
    settle(cell, .35)


def base_state():
    cell = Cell("lock")
    for name in ("p1a", "p1b"):
        set_part(cell, BY_NAME[name])
    for name in ("p1a", "p1b"):
        clamp(cell, BY_NAME[name])
    return cell


def trial(args):
    cell, seed, name = args
    rng = np.random.default_rng(seed)
    r, phi, dyaw = rng.uniform(0, MAX_ERROR_MM) * 1e-3, rng.uniform(0, 2 * np.pi), rng.uniform(-MAX_YAW, MAX_YAW)
    p = BY_NAME[name]
    state = cell.d.qpos.copy(), cell.d.ctrl.copy(), cell.d.time
    set_part(cell, p, r * np.cos(phi), r * np.sin(phi), dyaw, DROP)
    cell.d.qvel[:] = 0
    mujoco.mj_forward(cell.m, cell.d)
    settle(cell, .25)
    clamp(cell, p)
    e = cell.pose_errors()[name]
    cell.d.qpos[:], cell.d.ctrl[:], cell.d.time = state[0], state[1], state[2]
    cell.d.qvel[:] = 0
    mujoco.mj_forward(cell.m, cell.d)
    return {"part": name, "r_mm": r * 1e3, "yaw_mrad": dyaw * 1e3, "dx": e["dx_mm"], "dy": e["dy_mm"],
            "dz": e["dz_mm"], "yaw_after": e["yaw_mrad"]}


def worker(chunk):
    cell = None
    out = []
    for name, seeds in chunk:
        if name == "p1a":
            cell = Cell("lock")
        else:
            cell = base_state()
        out += [trial((cell, s, name)) for s in seeds]
    return out


def main():
    assert CTRL_DT > 0
    procs = 8
    seeds = np.arange(TRIALS)
    jobs = [[(n, seeds[k::procs] + (0 if n == "p1a" else 10_000)) for n in ("p1a", "p2a")] for k in range(procs)]
    with mp.Pool(procs) as pool:
        rows = [r for chunk in pool.map(worker, jobs) for r in chunk]
    with open(RESULTS / "capture.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for name in ("p1a", "p2a"):
        part = [r for r in rows if r["part"] == name]
        for lo in range(0, MAX_ERROR_MM, 2):
            b = [r for r in part if lo <= r["r_mm"] < lo + 2]
            # dz limit is wider: jaw friction leaves the part ~0.14 mm high, a top clamp has to remove it
            ok = [abs(r["dx"]) < .1 and abs(r["dy"]) < .1 and abs(r["dz"]) < .3 and abs(r["yaw_after"]) < 1 for r in b]
            print(name, f"{lo:2}-{lo + 2:2} mm n={len(b):3} seated={np.mean(ok):6.1%}")


if __name__ == "__main__":
    main()
