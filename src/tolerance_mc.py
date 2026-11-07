"""Monte Carlo tolerance analysis of hash assembly AMZ-SA-26-001 (all dimensions in mm)."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

L1, L2, W1, W2, PITCH = 320.0, 280.0, 40.0, 30.0, 120.0
N_SAMPLES = 200_000
STATIONS = np.linspace(-0.5, 0.5, 21)  # normalised positions along a profile
ROBOT_REPEATABILITY = 0.1  # datasheet value, read as a 3-sigma band
TOL = 0.05

REQUIREMENTS = (
    "120 Pos 1", "120 Pos 2", "100 (4x)", "60 (4x)",
    "parallel Pos 1", "perpendicular Pos 2", "parallel top", "gap",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    place_sigma: float        # in-plane placement error of a profile
    angle_sigma: float        # placement rotation [rad]
    length_tol: float         # half-width of the length tolerance
    width_tol: float          # half-width of the section width tolerance
    bow_max: float            # as-placed lateral and vertical bow per mm of length
    flank_located: bool       # one-sided stop: width error moves the axis
    centred_length: bool      # end error shared between ends, not pushed to one end
    leg_mismatch_max: float   # height step between the two legs of Pos 2
    twist_max: float          # [rad/mm]
    height_tol: float = 0.2   # half-width of the section height tolerance
    finished_after: bool = False  # Pos 2 ends and top machined in one setup after joining


GRIP_SIGMA = 0.2 / 3     # part-in-gripper uncertainty, not reduced by repeatability
LOCATOR_SIGMA = 0.01 / 3  # ground locator position
BOW_DELIVERED = 0.0017    # 0.17 % of length, secondary source for EN 10162
TWIST_DELIVERED = np.radians(1.0) / 1000
# The sagitta left between two stops scales with (span/length)^2; spans follow cad_fixture.py
POS1_MAX_SPAN = 60.0
POS2_MAX_SPAN = 66.0  # Pos 1 blocks the plate under Pos 2, stops sit next to its flanks
CLAMPED_BOW = BOW_DELIVERED * (POS1_MAX_SPAN / L1) ** 2
POS2_BOW_FACTOR = (POS2_MAX_SPAN / L2) ** 2 / (POS1_MAX_SPAN / L1) ** 2

SCENARIOS = (
    Scenario("Robot, ideal parts", ROBOT_REPEATABILITY / 3, ROBOT_REPEATABILITY / 3 / 150,
             0, 0, 0, False, False, 0, 0, 0),
    Scenario("Robot, delivered parts", np.hypot(ROBOT_REPEATABILITY / 3, GRIP_SIGMA),
             np.hypot(ROBOT_REPEATABILITY / 3, GRIP_SIGMA) / 150,
             0.5, 0.3, BOW_DELIVERED, False, False, 0.2, TWIST_DELIVERED),
    Scenario("Fixture, delivered parts", LOCATOR_SIGMA, LOCATOR_SIGMA / 150,
             0.5, 0.3, CLAMPED_BOW, True, False, 0.2, 0.2 * TWIST_DELIVERED),
    Scenario("Centring fixture, trimmed ends", LOCATOR_SIGMA, LOCATOR_SIGMA / 150,
             0.02, 0.3, CLAMPED_BOW, False, True, 0.2, 0.2 * TWIST_DELIVERED),
    Scenario("+ faced contact surfaces", LOCATOR_SIGMA, LOCATOR_SIGMA / 150,
             0.02, 0.3, CLAMPED_BOW, False, True, 0.02, 0.02 * TWIST_DELIVERED, 0.01),
    Scenario("+ finish machining after joining", LOCATOR_SIGMA, LOCATOR_SIGMA / 150,
             0.02, 0.3, CLAMPED_BOW, False, True, 0.02, 0.02 * TWIST_DELIVERED, 0.01, True),
)
FINISH_TOL = 0.02  # three-axis milling in one setup, referenced to the assembled frame


def bounded(rng, half_width, shape, dist):
    if dist == "uniform":
        return rng.uniform(-half_width, half_width, shape)
    return np.clip(rng.normal(0, half_width / 3, shape), -half_width, half_width)


def profile_flank(position, angle, bow, length):
    """Lateral flank coordinate along a profile, shape (samples, stations)."""
    s = STATIONS * length
    return position[:, None] + angle[:, None] * s + bow[:, None] * (1 - (2 * STATIONS) ** 2)


def peak_to_valley(values):
    return values.max(axis=1) - values.min(axis=1)


def simulate(sc, n=N_SAMPLES, dist="normal", seed=1):
    rng = np.random.default_rng(seed)
    length = np.stack([L1 + bounded(rng, sc.length_tol, n, dist) for _ in range(2)]
                      + [L2 + bounded(rng, sc.length_tol, n, dist) for _ in range(2)], axis=1)
    width = np.stack([W1 + bounded(rng, sc.width_tol, n, dist) for _ in range(2)]
                     + [W2 + bounded(rng, 0.2 if sc.width_tol else 0, n, dist) for _ in range(2)], axis=1)
    ref_width = np.array([W1, W1, W2, W2])
    nominal = np.array([-PITCH / 2, PITCH / 2, -PITCH / 2, PITCH / 2])

    axis = nominal + rng.normal(0, sc.place_sigma, (n, 4))
    if sc.flank_located:
        axis += (width - ref_width) / 2
    along = rng.normal(0, sc.place_sigma, (n, 4))
    if sc.flank_located and not sc.centred_length:
        # end stop on one side: the length error appears at the free end only
        along += (length - np.array([L1, L1, L2, L2])) / 2
    angle = rng.normal(0, sc.angle_sigma, (n, 4))
    # Pos 1 occupies the plate under Pos 2, so its stops cannot be evenly spaced
    k = POS2_BOW_FACTOR if sc.bow_max == CLAMPED_BOW else 1.0
    bow = rng.uniform(0, 1, (n, 4)) * sc.bow_max * np.array([L1, L1, k * L2, k * L2])
    camber = rng.uniform(0, 1, (n, 2)) * sc.bow_max * L2

    res = {}
    res["120 Pos 1"] = np.abs(axis[:, 1] - axis[:, 0] - PITCH)
    res["120 Pos 2"] = np.abs(axis[:, 3] - axis[:, 2] - PITCH)

    end_low = along[:, :2] - length[:, :2] / 2
    end_high = along[:, :2] + length[:, :2] / 2
    res["100 (4x)"] = np.max(np.abs(np.stack([
        axis[:, 2] - end_low[:, 0] - (L1 / 2 - PITCH / 2),
        axis[:, 2] - end_low[:, 1] - (L1 / 2 - PITCH / 2),
        end_high[:, 0] - axis[:, 3] - (L1 / 2 - PITCH / 2),
        end_high[:, 1] - axis[:, 3] - (L1 / 2 - PITCH / 2),
    ], axis=1)), axis=1)

    end2_low = along[:, 2:] - length[:, 2:] / 2
    end2_high = along[:, 2:] + length[:, 2:] / 2
    flank_low = axis[:, 0] - width[:, 0] / 2
    flank_high = axis[:, 1] + width[:, 1] / 2
    res["60 (4x)"] = np.max(np.abs(np.stack([
        flank_low - end2_low[:, 0] - 60, flank_low - end2_low[:, 1] - 60,
        end2_high[:, 0] - flank_high - 60, end2_high[:, 1] - flank_high - 60,
    ], axis=1)), axis=1)

    datum = angle[:, 0]  # lower Pos 1 is the datum for both orientation requirements
    flank1 = profile_flank(np.zeros(n), angle[:, 1] - datum, bow[:, 1], L1)
    res["parallel Pos 1"] = peak_to_valley(flank1)
    res["perpendicular Pos 2"] = np.maximum(
        peak_to_valley(profile_flank(np.zeros(n), angle[:, 2] - datum, bow[:, 2], L2)),
        peak_to_valley(profile_flank(np.zeros(n), angle[:, 3] - datum, bow[:, 3], L2)))

    # Pos 2 spans both Pos 1, so their height difference is amplified by 280/120 at the tips
    h1 = bounded(rng, sc.height_tol, (n, 2), dist)
    h2 = bounded(rng, sc.height_tol, (n, 2), dist)
    x = STATIONS[None, None, :] * L2
    top_z = h1.mean(axis=1)[:, None, None] \
        + ((h1[:, 1] - h1[:, 0]) / PITCH)[:, None, None] * x \
        + h2[:, :, None] + camber[:, :, None] * (1 - (2 * STATIONS[None, None, :]) ** 2)
    res["parallel top"] = top_z.max(axis=(1, 2)) - top_z.min(axis=(1, 2))

    leg_step = rng.uniform(0, 1, (n, 2)) * sc.leg_mismatch_max
    twist_gap = rng.uniform(0, 1, (n, 2)) * sc.twist_max * PITCH * W2
    res["gap"] = (leg_step + twist_gap).max(axis=1)
    if sc.finished_after:
        for key in ("100 (4x)", "60 (4x)", "parallel top"):
            res[key] = np.abs(bounded(rng, FINISH_TOL, n, dist))
    return res


def yields(res):
    per_req = {k: float(np.mean(v <= TOL)) for k, v in res.items()}
    all_ok = np.logical_and.reduce([v <= TOL for v in res.values()])
    return per_req, float(np.mean(all_ok))


def self_check():
    zero = Scenario("zero", 0, 0, 0, 0, 0, False, False, 0, 0, 0)
    per_req, total = yields(simulate(zero, n=100))
    assert total == 1.0 and all(v == 1.0 for v in per_req.values())
    # length chain: a 0.2 mm long Pos 1 in a centred setup puts each 100 mm off by 0.1
    only_length = Scenario("len", 0, 0, 0.5, 0, 0, False, True, 0, 0, 0)
    res = simulate(only_length, n=100_000, dist="uniform")
    assert 0.17 < np.mean(res["100 (4x)"] <= TOL) ** 0.5 < 0.23


def main():
    self_check()
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for dist in ("normal", "uniform"):
        for sc in SCENARIOS:
            per_req, total = yields(simulate(sc, dist=dist))
            rows.append({"distribution": dist, "scenario": sc.name, "all": total, **per_req})
    with open(out / "tolerance_yield.csv", "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['distribution']:8}{r['scenario']:27}all={r['all']:7.2%}  "
              + " ".join(f"{r[k]:6.1%}" for k in REQUIREMENTS))


if __name__ == "__main__":
    main()
