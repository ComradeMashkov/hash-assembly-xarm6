"""Kinematic control of both arms: resolved-rate tracking plus a control barrier function filter."""
import mujoco
import numpy as np

ARM_DOF = 6
SEGMENT_RADIUS = (.065, .06, .055)  # upper arm, forearm, wrist and tool
PART_RADIUS = .02
D_SAFE = .03


def tool_rotation(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0])
    z = np.array([0, 0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def segment_closest(a0, a1, b0, b1):
    """Closest points of two 3D segments (Ericson, Real-Time Collision Detection, 5.1.9)."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    aa, ee, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = aa * ee - b * b
    s = np.clip((b * f - c * ee) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / ee
    if t < 0:
        t, s = 0.0, np.clip(-c / aa, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / aa, 0, 1)
    return a0 + d1 * s, b0 + d2 * t


class Arm:
    def __init__(self, model, data, i):
        self.m, self.d, self.i = model, data, i
        self.site = model.site(f"r{i}_tcp").id
        names = [f"r{i}_j{k}" for k in range(1, 7)]
        self.qadr = np.array([model.joint(n).qposadr[0] for n in names])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in names])
        self.act = np.array([model.actuator(f"r{i}_a{k}").id for k in range(1, 7)])
        self.seg_bodies = [model.body(f"r{i}_{n}").id for n in ("link2", "link3", "link5", "tool")]
        self.q_cmd = data.qpos[self.qadr].copy()
        self.held = None   # extent of a carried part along the tool x axis

    def q(self):
        return self.d.qpos[self.qadr]

    def tcp(self):
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self, point=None, body=None):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        if point is None:
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        else:
            mujoco.mj_jac(self.m, self.d, jp, jr, point, body)
        return jp[:, self.vadr], jr[:, self.vadr]

    def resolved_rate(self, pos, rot, gain=6.0, vmax=.35):
        p, r = self.tcp()
        ev = pos - p
        n = np.linalg.norm(ev)
        v = gain * ev
        if gain * n > vmax:
            v *= vmax / (gain * n)
        err = r.T @ rot
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, err.flatten())
        w = np.zeros(3)
        mujoco.mju_quat2Vel(w, quat, 1.0)
        w = r @ w
        jp, jr = self.jacobian()
        J = np.vstack([jp, jr])
        return np.clip(J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[v, 3.0 * w]), -2.0, 2.0)

    def segments(self):
        pts = [self.d.xpos[b] for b in self.seg_bodies]
        tip, rot = self.tcp()
        segs = [(pts[0], pts[1], self.seg_bodies[0], SEGMENT_RADIUS[0]),
                (pts[1], pts[2], self.seg_bodies[1], SEGMENT_RADIUS[1]),
                (pts[2], tip, self.seg_bodies[2], SEGMENT_RADIUS[2])]
        if self.held:
            axis = rot[:, 0]
            segs.append((tip + axis * self.held[0], tip + axis * self.held[1], self.seg_bodies[3], PART_RADIUS))
        return segs


def barrier_constraints(a, b, alpha, nv=2 * ARM_DOF):
    """Rows G qdot >= g of h_dot + alpha h >= 0, with h the clearance of each capsule pair."""
    G, g, hmin = [], [], np.inf
    for pa, qa, ba, ra in a.segments():
        for pb, qb, bb, rb in b.segments():
            ca, cb = segment_closest(pa, qa, pb, qb)
            delta = ca - cb
            dist = np.linalg.norm(delta)
            clearance = dist - ra - rb
            hmin = min(hmin, clearance)
            h = clearance - D_SAFE
            if h > .2 or dist < 1e-9:
                continue
            n = delta / dist
            row = np.zeros(nv)
            row[:ARM_DOF] = n @ a.jacobian(ca, ba)[0]
            row[ARM_DOF:] = -n @ b.jacobian(cb, bb)[0]
            G.append(row)
            g.append(-alpha * h)
    return np.array(G).reshape(-1, nv), np.array(g), hmin


def cbf_filter(u, G, g, weight, iterations=60):
    """min (x-u)'W(x-u) s.t. Gx >= g, solved by Hildreth's dual coordinate ascent."""
    if not len(g):
        return u
    Winv = 1 / weight
    H = (G * Winv) @ G.T
    s = g - G @ u
    lam = np.zeros(len(g))
    for _ in range(iterations):
        for i in range(len(g)):
            lam[i] = max(0.0, lam[i] + (s[i] - H[i] @ lam) / max(H[i, i], 1e-9))
    return u + Winv * (G.T @ lam)


if __name__ == "__main__":
    a, b = segment_closest(*map(np.array, ([0., 0, 0], [1., 0, 0], [.5, 1, 0], [.5, 2, 0])))
    assert np.allclose(a, [.5, 0, 0]) and np.allclose(b, [.5, 1, 0])
    G, g = np.array([[1.0, 0.0]]), np.array([1.0])
    x = cbf_filter(np.array([0.0, 2.0]), G, g, np.ones(2))
    assert np.allclose(x, [1.0, 2.0])
    print("control self-check ok")
