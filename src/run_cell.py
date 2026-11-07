"""Runs the assembly cycle in simulation. Modes: lock (zone lock baseline) and cbf (barrier filter)."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from cell import BASE_Y, GRIP, PARTS, build_xml
from control import Arm, barrier_constraints, cbf_filter, tool_rotation

RESULTS = Path(__file__).resolve().parents[1] / "results"
CTRL_DT = .01
SAFE_Z = .16      # transit height of the TCP
DROP = .012       # release height above the seat
SIDE_CLAMP_DELAY = .4  # ends centre first: side clamps would lock the part by friction


def tcp_height(p, seated):
    base = p.z if seated else 0.0
    return base + p.height + (DROP if seated else 0.0)


def task_list(robot, bias):
    """Waypoints for one robot: (name, pos, yaw, action, part, shared_zone)."""
    steps = []
    home = np.array([.0, np.sign(robot - .5) * (BASE_Y - .17), SAFE_Z + .06])
    for p in [q for q in PARTS if q.robot == robot]:
        tx, ty, tyaw = p.tray
        axis = np.array([np.cos(p.yaw), np.sin(p.yaw)])
        tx, ty = np.array([tx, ty]) + axis * GRIP[p.name]
        e = bias.get(p.name, (0, 0, 0))
        place = np.array([p.x + e[0], p.y + e[1]]) + axis * GRIP[p.name]
        yaw = p.yaw + e[2]
        steps += [("above tray", [tx, ty, SAFE_Z], tyaw, None, p, False),
                  ("pick", [tx, ty, p.height], tyaw, "grip", p, False),
                  ("lift", [tx, ty, SAFE_Z], tyaw, None, p, False),
                  ("above seat", [*place, SAFE_Z], yaw, None, p, True),
                  ("place", [*place, tcp_height(p, True)], yaw, "release", p, True),
                  ("retreat", [*place, SAFE_Z], yaw, "seat", p, True),
                  ("clear", list(home), 0.0, "done", p, False)]
    return steps


class Cell:
    def __init__(self, mode, bias=None, alpha=4.0, record=None):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetData(self.m, self.d)
        self.mode, self.alpha, self.record = mode, alpha, record
        self.arms = [Arm(self.m, self.d, i) for i in (0, 1)]
        bias = bias or {}
        self.tasks = [task_list(i, bias) for i in (0, 1)]
        self.index = [0, 0]
        self.wait = [0.0, 0.0]
        self.seated = set()
        self.side_clamp = []
        self.correction = []
        self.dropped = {}
        self.lock = None
        self.min_clearance = np.inf
        self.frames, self.log = [], []
        self.renderer = mujoco.Renderer(self.m, 720, 1280) if record else None
        for a, home in zip(self.arms, ([0, -BASE_Y + .17, .22], [0, BASE_Y - .17, .22])):
            self.ik(a, np.array(home), 0.0)
        mujoco.mj_forward(self.m, self.d)

    def ik(self, arm, pos, yaw, iterations=800):
        arm.q_cmd = arm.q().copy()
        arm.q_cmd[[1, 2, 4]] = [-.3, -.9, 1.2]
        self.d.qpos[arm.qadr] = arm.q_cmd
        for _ in range(iterations):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            self.d.qpos[arm.qadr] += .05 * arm.resolved_rate(pos, tool_rotation(yaw), vmax=2.0)
        arm.q_cmd = arm.q().copy()
        self.d.ctrl[arm.act] = arm.q_cmd

    def grip(self, arm, p, on):
        eq = self.m.equality(f"grip_{p.name}").id
        if on:
            tool, body = self.m.body(f"r{arm.i}_tool").id, self.m.body(p.name).id
            r1 = self.d.xmat[tool].reshape(3, 3)
            rel = r1.T @ (self.d.xpos[body] - self.d.xpos[tool])
            q1, q2, inv, rq = self.d.xquat[tool], self.d.xquat[body], np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, q1)
            mujoco.mju_mulQuat(rq, inv, q2)
            self.m.eq_data[eq, 3:6] = rel
            self.m.eq_data[eq, 6:10] = rq
            arm.held = (-p.length / 2 - GRIP[p.name], p.length / 2 - GRIP[p.name])
        else:
            arm.held = None
        self.d.eq_active[eq] = on

    def blocked(self, i, step):
        """Zone lock and assembly precedence; returns True while the robot has to wait."""
        name, _, _, _, p, shared = step
        if name == "above seat" and p.name.startswith("p2") and \
                not {"p1a", "p1b"} <= self.seated:
            return True
        if self.mode == "lock" and shared:
            if self.lock not in (None, i):
                return True
            self.lock = i
        return False

    def control_step(self):
        u = np.zeros(12)
        for i, arm in enumerate(self.arms):
            if self.index[i] >= len(self.tasks[i]):
                continue
            step = self.tasks[i][self.index[i]]
            name, pos, yaw, action, p, shared = step
            if self.wait[i] > 0:
                self.wait[i] -= CTRL_DT
                continue
            if self.blocked(i, step):
                continue
            pos = np.array(pos, float)
            u[6 * i:6 * i + 6] = arm.resolved_rate(pos, tool_rotation(yaw))
            near = np.linalg.norm(arm.tcp()[0] - pos) < .0015
            if near:
                u[6 * i:6 * i + 6] = 0
                if action == "grip":
                    self.grip(arm, p, True)
                    self.wait[i] = .25
                elif action == "release":
                    self.grip(arm, p, False)
                    self.wait[i] = .4
                elif action == "seat":
                    self.seat(p)
                    self.wait[i] = .2
                elif action == "done" and self.lock == i:
                    self.lock = None
                self.index[i] += 1
                if self.mode == "lock" and name == "retreat":
                    self.lock = None
        G, g, hmin = barrier_constraints(*self.arms, self.alpha)
        self.min_clearance = min(self.min_clearance, hmin)
        if self.mode == "cbf":
            weight = np.r_[np.ones(6), 1.5 * np.ones(6)]  # asymmetric weights break head-on deadlocks
            nominal = u
            u = cbf_filter(u, G, g, weight)
            self.correction.append(float(np.linalg.norm(u - nominal)))
        for i, arm in enumerate(self.arms):
            arm.q_cmd += np.clip(u[6 * i:6 * i + 6], -2.5, 2.5) * CTRL_DT
            self.d.ctrl[arm.act] = arm.q_cmd
        self.log.append((self.d.time, hmin))

    def seat(self, p):
        self.seated.add(p.name)
        self.close_jaws(p, "e")
        self.side_clamp.append((self.d.time + SIDE_CLAMP_DELAY, p))

    def close_jaws(self, p, tag):
        # force-limited closing, both jaws of a pair move inward
        self.d.ctrl[self.m.actuator(f"{p.name}_{tag}n").id] = .02
        self.d.ctrl[self.m.actuator(f"{p.name}_{tag}p").id] = -.02

    def update_side_clamps(self):
        for t, p in [x for x in self.side_clamp if x[0] <= self.d.time]:
            self.close_jaws(p, "s")
            self.side_clamp.remove((t, p))

    def finished(self):
        return all(self.index[i] >= len(self.tasks[i]) for i in (0, 1))

    def run(self, t_max=90.0):
        sub = int(CTRL_DT / self.m.opt.timestep)
        next_frame = 0.0
        while self.d.time < t_max and not self.finished():
            self.control_step()
            self.update_side_clamps()
            mujoco.mj_step(self.m, self.d, nstep=sub)
            if self.renderer and self.d.time >= next_frame:
                self.renderer.update_scene(self.d, camera="front")
                self.frames.append(self.renderer.render())
                next_frame += 1 / 30
        cycle = self.d.time
        for _ in range(100):
            self.update_side_clamps()
            mujoco.mj_step(self.m, self.d, nstep=sub)
        return cycle

    def pose_errors(self):
        out = {}
        for p in PARTS:
            pos = self.d.xpos[self.m.body(p.name).id]
            mat = self.d.xmat[self.m.body(p.name).id].reshape(3, 3)
            axis = mat[:, 0]
            nominal = np.array([p.x, p.y, p.z + p.height / 2])
            yaw_error = np.arctan2(axis[1], axis[0]) - p.yaw
            out[p.name] = {"dx_mm": float(1e3 * (pos[0] - nominal[0])),
                           "dy_mm": float(1e3 * (pos[1] - nominal[1])),
                           "dz_mm": float(1e3 * (pos[2] - nominal[2])),
                           "yaw_mrad": float(1e3 * yaw_error)}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lock", "cbf"), default="cbf")
    ap.add_argument("--video")
    ap.add_argument("--t-max", type=float, default=90)
    a = ap.parse_args()
    cell = Cell(a.mode, record=a.video)
    cycle = cell.run(a.t_max)
    result = {"mode": a.mode, "finished": cell.finished(), "cycle_s": cycle,
              "min_clearance_mm": 1e3 * cell.min_clearance,
              "filter_active_share": float(np.mean(np.array(cell.correction) > 1e-3)) if cell.correction else 0.0,
              "filter_max_correction_rad_s": max(cell.correction, default=0.0), "seated": sorted(cell.seated),
              "errors": cell.pose_errors()}
    print(json.dumps(result, indent=1))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"cycle_{a.mode}.json").write_text(json.dumps(result, indent=1))
    np.save(RESULTS / f"clearance_{a.mode}.npy", np.array(cell.log))
    if a.video:
        imageio.mimsave(a.video, cell.frames, fps=30, quality=8, macro_block_size=8)


if __name__ == "__main__":
    main()
