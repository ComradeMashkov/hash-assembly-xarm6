"""Work cell definition: builds the MuJoCo model of two xArm6 and the centring fixture (SI units)."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / "third_party/xarm_ros2/xarm_description/meshes/xarm6/visual"

PITCH = 0.120
BASE_Y = 0.40        # robots face each other across the fixture
TRAY = (0.38, 0.38)  # pick positions, mirrored through the cell centre for R2
JAW_OPEN = 0.004     # clearance per side while the part drops in
FINGER_H = 0.004     # low fingers: legs enter them after a short drop
CHAMFER = 0.007      # lead-in width per side, a 45 deg ramp rising away from the part
WALL = 0.002
Z1, Z2 = 0.020, 0.015  # section heights


@dataclass(frozen=True)
class Part:
    name: str
    length: float
    width: float
    height: float
    x: float       # nominal seated centre
    y: float
    z: float       # height of the open side
    yaw: float     # 0: along x, 90 deg: along y
    robot: int
    tray: tuple    # (x, y, yaw) pick pose


TRAY_R = {0: (TRAY[0], -TRAY[1]), 1: (-TRAY[0], TRAY[1])}
PARTS = (
    Part("p1a", .320, .040, Z1, 0, -PITCH / 2, 0, 0, 0, (*TRAY_R[0], 0)),
    Part("p1b", .320, .040, Z1, 0, PITCH / 2, 0, 0, 1, (*TRAY_R[1], 0)),
    Part("p2a", .280, .030, Z2, -PITCH / 2, 0, Z1, 1.5708, 0, (-TRAY[0], -TRAY[1], 1.5708)),
    Part("p2b", .280, .030, Z2, PITCH / 2, 0, Z1, 1.5708, 1, (TRAY[0], TRAY[1], 1.5708)),
)

# Offset of the grip along the part axis: wrists do not fit side by side at the 120 mm pitch.
GRIP = {"p1a": -.08, "p1b": .08, "p2a": -.08, "p2b": .08}

LINKS = [  # name, parent offset, roll, mass, joint range
    ("link1", (0, 0, .267), 0, 2.4, (-3.11, 3.11)),
    ("link2", (0, 0, 0), -1.5708, 2.6, (-2.059, 2.0944)),
    ("link3", (.0535, -.2845, 0), 0, 2.2, (-3.11, .19198)),
    ("link4", (.0775, .3425, 0), -1.5708, 1.3, (-3.11, 3.11)),
    ("link5", (0, 0, 0), 1.5708, 1.5, (-1.69297, 3.11)),
    ("link6", (.076, .097, 0), -1.5708, .2, (-3.11, 3.11)),
]


def quat_roll(r):
    import math
    return f"{math.cos(r / 2):.6f} {math.sin(r / 2):.6f} 0 0"


def arm_xml(i, base_y, yaw_quat, rgba):
    closing, body = "", f'<body name="r{i}_base" pos="0 {base_y} 0" quat="{yaw_quat}">' \
        f'<geom type="mesh" mesh="link_base" rgba="{rgba}" contype="0" conaffinity="0"/>'
    for name, pos, roll, mass, rng in LINKS:
        body += (f'<body name="r{i}_{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat_roll(roll)}" gravcomp="1">'
                 f'<inertial pos="0 0 0" mass="{mass}" diaginertia=".01 .01 .01"/>'
                 f'<joint name="r{i}_j{name[-1]}" axis="0 0 1" range="{rng[0]} {rng[1]}" damping="20" armature=".1"/>'
                 f'<geom type="mesh" mesh="{name}" rgba="{rgba}" contype="0" conaffinity="0"/>')
        closing += "</body>"
    body += (f'<body name="r{i}_tool" pos="0 0 0" gravcomp="1">'
             '<inertial pos="0 0 .05" mass=".3" diaginertia=".001 .001 .001"/>'
             '<geom type="cylinder" size=".022 .03" pos="0 0 .03" rgba=".25 .27 .3 1" contype="0" conaffinity="0"/>'
             '<geom type="box" size=".09 .012 .008" pos="0 0 .068" rgba=".25 .27 .3 1" contype="0" conaffinity="0"/>'
             '<geom type="cylinder" size=".01 .003" pos=".07 0 .077" rgba=".1 .1 .1 1" contype="0" conaffinity="0"/>'
             '<geom type="cylinder" size=".01 .003" pos="-.07 0 .077" rgba=".1 .1 .1 1" contype="0" conaffinity="0"/>'
             f'<site name="r{i}_tcp" pos="0 0 .08" size=".004"/></body>')
    return body + closing + "</body>"


def profile_xml(p):
    q = "0.707107 0 0 0.707107" if p.yaw else "1 0 0 0"
    hl, hw, hh = p.length / 2, p.width / 2, p.height / 2
    rgba = ".62 .66 .7 1" if p.name.startswith("p1") else ".74 .68 .55 1"
    legs = "".join(f'<geom type="box" size="{hl} {WALL / 2} {hh}" pos="0 {s * (hw - WALL / 2)} 0" rgba="{rgba}" class="metal"/>'
                   for s in (-1, 1))
    tx, ty, tyaw = p.tray
    tq = "0.707107 0 0 0.707107" if tyaw else "1 0 0 0"
    return (f'<body name="{p.name}" pos="{tx} {ty} {hh + .001}" quat="{tq}">'
            f'<freejoint name="{p.name}_free"/>{legs}'
            f'<geom type="box" size="{hl} {hw} {WALL / 2}" pos="0 0 {hh - WALL / 2}" rgba="{rgba}" class="metal"/>'
            f'<inertial pos="0 0 {hh * .4:.4f}" mass="{.38 if p.length > .3 else .25}" '
            f'diaginertia=".003 .003 .006"/></body>')


def jaw(name, axis, pos, fingers, inward):
    """One moving jaw: slides along `axis`, each finger is a low block with a 45 deg lead-in."""
    fx = "".join(
        f'<geom type="box" size="{sx} {sy} {sz}" pos="{ox} {oy} {sz}" rgba=".2 .34 .52 1" class="metal"/>'
        f'<geom type="box" size="{sx} {sy} .0005" '
        f'pos="{ox + inward[0] * .0012:.4f} {oy + inward[1] * .0012:.4f} {2 * sz + .0035}" '
        f'euler="{-inward[1] * .7854} {inward[0] * .7854} 0" rgba=".35 .5 .68 1" class="metal"/>'
        for sx, sy, sz, ox, oy in fingers)
    return (f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">'
            f'<joint name="{name}" type="slide" axis="{axis}" range="-.02 .02" damping="300"/>'
            f'<inertial pos="0 0 0" mass=".2" diaginertia=".001 .001 .001"/>{fx}</body>')


def fixture_xml():
    bodies, eqs, acts = "", "", ""
    h = FINGER_H / 2
    for p in PARTS:
        c = (p.x, p.y)
        along = 1 if p.yaw else 0   # index of the long axis
        cross = 1 - along
        step = .115 if p.name.startswith("p1") else .110
        for label, ax, half, spots in (("s", cross, p.width / 2 + JAW_OPEN, (-step, step)),
                                       ("e", along, p.length / 2 + JAW_OPEN,  # an open section: end jaws must meet the legs
                                        (-(p.width / 2 - WALL / 2), p.width / 2 - WALL / 2))):
            axis = "1 0 0" if ax == 0 else "0 1 0"
            for sign, tag in ((-1, "n"), (1, "p")):
                pos = [c[0], c[1], p.z]
                pos[ax] += sign * (half + .005)
                fingers = []
                for sp in spots:
                    off = [0, 0]
                    off[1 - ax] = sp
                    size = [.005, .005]
                    size[1 - ax] = .006
                    fingers.append((size[0], size[1], h, off[0], off[1]))
                inward = (-sign if ax == 0 else 0, -sign if ax == 1 else 0)
                bodies += jaw(f"{p.name}_{label}{tag}", axis, pos, fingers, inward)
            eqs += (f'<joint joint1="{p.name}_{label}n" joint2="{p.name}_{label}p" polycoef="0 -1 0 0 0" '
                    f'solref=".002 1" solimp=".99 .999 .0005"/>')
            for tag in "np":
                acts += (f'<position name="{p.name}_{label}{tag}" joint="{p.name}_{label}{tag}" kp="20000" '
                         f'forcerange="-150 150" ctrlrange="-.02 .02"/>')
    return bodies, eqs, acts


def build_xml():
    meshes = "".join(f'<mesh name="{n}" file="{(MESHES / (n + ".stl")).as_posix()}"/>'
                     for n in ("link_base", *(l[0] for l in LINKS)))
    jaws, eqs, jaw_acts = fixture_xml()
    arm_acts = "".join(f'<position name="r{i}_a{k}" joint="r{i}_j{k}" kp="4000" dampratio="1" '
                       f'forcerange="-120 120"/>' for i in (0, 1) for k in range(1, 7))
    welds = "".join(f'<weld name="grip_{p.name}" body1="r{p.robot}_tool" body2="{p.name}" active="false"/>'
                    for p in PARTS)
    return f"""<mujoco model="hash_cell">
<compiler angle="radian" autolimits="true"/>
<option timestep=".001" integrator="implicitfast" cone="elliptic" impratio="5"/>
<visual><global offwidth="1600" offheight="900"/><quality shadowsize="4096"/></visual>
<default><default class="metal"><geom condim="3" friction=".25" solref=".004 1" solimp=".95 .99 .0005"/></default></default>
<asset>{meshes}
<texture type="skybox" builtin="gradient" rgb1=".93 .94 .95" rgb2=".78 .8 .83" width="64" height="64"/><texture name="grid" type="2d" builtin="checker" rgb1=".86 .87 .88" rgb2=".82 .83 .84" width="256" height="256"/>
<material name="plate" texture="grid" texrepeat="48 48"/></asset>
<worldbody>
<light pos=".5 -.8 2.2" dir="-.2 .3 -1" diffuse=".55 .55 .55" specular=".1 .1 .1" castshadow="true"/><light pos="-.8 .8 2" dir=".3 -.3 -1" diffuse=".4 .4 .4" specular="0 0 0" castshadow="false"/>
<geom name="plate" type="plane" size="1.6 1.6 .01" material="plate" class="metal"/>
<camera name="front" pos=".82 -.98 .88" xyaxes=".77 .64 0 -.39 .47 .79"/>
<camera name="top" pos="0 -.25 1.15" xyaxes="1 0 0 0 .97 .22"/>
{arm_xml(0, -BASE_Y, ".707107 0 0 .707107", ".93 .93 .94 1")}
{arm_xml(1, BASE_Y, ".707107 0 0 -.707107", ".85 .88 .93 1")}
{jaws}{''.join(profile_xml(p) for p in PARTS)}
</worldbody>
<equality>{eqs}{welds}</equality>
<actuator>{arm_acts}{jaw_acts}</actuator>
</mujoco>"""
