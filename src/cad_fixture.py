"""CadQuery concept model of the centring fixture with the assembled hash (mm). Exports STEP and SVG."""
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parents[1] / "cad"
OUT.mkdir(exist_ok=True)

PITCH = 120
PLATE = (480, 440, 20)
POS1_STOPS = (-155, -110, -60, 0, 60, 110, 155)   # along x, working on the 12 mm below the web
POS2_STOPS = (-135, -87, -33, 33, 87, 135)        # along y, free plate next to the Pos 1 flanks
PAD = 12
LOW, HIGH = 12, 32   # stop heights: Pos 1 flank, Pos 2 flank (on pillars)


def u_profile(length, width, height, wall=2):
    outer = cq.Workplane("YZ").rect(width, height).extrude(length / 2, both=True)
    pocket = cq.Workplane("YZ").center(0, -wall).rect(width - 2 * wall, height - wall + wall).extrude(length, both=True)
    return outer.cut(pocket).translate((0, 0, height / 2))


def block(x0, x1, y0, y1, height):
    return cq.Workplane("XY").box(abs(x1 - x0), abs(y1 - y0), height, centered=(False, False, False)) \
        .translate((min(x0, x1), min(y0, y1), 0))


asm = cq.Assembly(name="hash_fixture")
asm.add(block(-PLATE[0] / 2, PLATE[0] / 2, -PLATE[1] / 2, PLATE[1] / 2, PLATE[2]).translate((0, 0, -PLATE[2])),
        name="base_plate", color=cq.Color(.3, .34, .4))
steel = cq.Color(.72, .74, .77)
for s in (-1, 1):
    asm.add(u_profile(320, 40, 20).translate((0, s * PITCH / 2, 0)), name=f"pos1_{s}", color=steel)
    asm.add(u_profile(280, 30, 15).rotate((0, 0, 0), (0, 0, 1), 90).translate((s * PITCH / 2, 0, 20)),
            name=f"pos2_{s}", color=steel)

blue, light = cq.Color(.17, .36, .6), cq.Color(.25, .46, .7)
for s in (-1, 1):
    y_outer, y_inner = s * 80, s * 40
    for x in POS1_STOPS:
        asm.add(block(x - PAD / 2, x + PAD / 2, y_outer, y_outer + s * 30, LOW), name=f"p1_out_{s}_{x}", color=blue)
        asm.add(block(x - PAD / 2, x + PAD / 2, y_inner, y_inner - s * 14, LOW), name=f"p1_in_{s}_{x}", color=blue)
    x_outer, x_inner = s * 75, s * 45
    for y in POS2_STOPS:
        asm.add(block(x_outer, x_outer + s * 16, y - PAD / 2, y + PAD / 2, HIGH), name=f"p2_out_{s}_{y}", color=light)
        asm.add(block(x_inner, x_inner - s * 16, y - PAD / 2, y + PAD / 2, HIGH), name=f"p2_in_{s}_{y}", color=light)
    # End stops act on the legs: the open section would swallow a central stop
    for row in (-1, 1):
        for leg in (-19, 19):
            y = row * PITCH / 2 + leg
            asm.add(block(s * 160, s * 190, y - PAD / 2, y + PAD / 2, LOW), name=f"p1_end_{s}_{row}_{leg}", color=blue)
            x = row * PITCH / 2 + leg * 14 / 19
            asm.add(block(x - PAD / 2, x + PAD / 2, s * 140, s * 170, HIGH), name=f"p2_end_{s}_{row}_{leg}", color=light)

asm.export(str(OUT / "hash_fixture.step"))
cq.exporters.export(cq.Workplane(obj=asm.toCompound()), str(OUT / "hash_fixture.svg"), opt={
    "width": 1100, "height": 560, "projectionDir": (.8, -1.0, .75), "showHidden": False,
    "strokeWidth": .3, "showAxes": False})
