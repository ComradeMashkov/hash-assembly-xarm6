"""Side-by-side video: zone lock vs. barrier filter, with a running clock."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parents[1] / "media"
FPS = 30
W, H = 960, 540


def load(name):
    return [np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS)) for f in imageio.mimread(MEDIA / name, memtest=False)]


def label(frame, title, t, done):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)
    small = ImageFont.truetype("arial.ttf", 24)
    d.rectangle((0, 0, W, 52), fill=(25, 32, 42))
    d.text((18, 10), title, font=font, fill=(255, 255, 255))
    d.text((W - 150, 13), f"{t:5.1f} s", font=small, fill=(255, 255, 255))
    if done:
        d.text((18, H - 44), f"cycle complete: {done:.1f} s", font=small, fill=(20, 60, 110))
    return np.asarray(img)


lock, cbf = load("cycle_lock.mp4"), load("cycle_cbf.mp4")
n = max(len(lock), len(cbf)) + FPS * 2
writer = imageio.get_writer(MEDIA / "comparison.mp4", fps=FPS, quality=8, macro_block_size=8)
for k in range(n):
    t = k / FPS
    left = label(lock[min(k, len(lock) - 1)], "Zone lock", min(t, len(lock) / FPS), len(lock) / FPS if k >= len(lock) else 0)
    right = label(cbf[min(k, len(cbf) - 1)], "Barrier filter", min(t, len(cbf) / FPS), len(cbf) / FPS if k >= len(cbf) else 0)
    writer.append_data(np.hstack([left, np.full((H, 6, 3), 255, np.uint8), right]))
writer.close()
print("frames", n, "lock", len(lock) / FPS, "cbf", len(cbf) / FPS)
