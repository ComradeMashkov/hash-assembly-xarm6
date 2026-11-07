"""Which touch points make the transform between two robot bases observable?

A touch of a plane with normal n at point p measures n'(R p + t). Linearised for a small pose
error, each touch gives the row [n, p x n]; the transform is observable only at rank 6.
"""
import numpy as np


def rows(touches):
    return np.array([np.r_[n, np.cross(p, n)] for p, n in touches], float)


Z, Y, X = (0, 0, 1), (0, 1, 0), (1, 0, 0)
PLATE_ONLY = [((.2, .2, 0), Z), ((-.2, .2, 0), Z), ((0, -.2, 0), Z), ((.1, 0, 0), Z), ((-.1, -.1, 0), Z), ((0, .1, 0), Z)]
DATUM_321 = [((.2, .2, 0), Z), ((-.2, .2, 0), Z), ((0, -.2, 0), Z),
             ((.15, -.22, -.01), Y), ((-.15, -.22, -.01), Y), ((.24, 0, -.01), X)]

for name, touches in (("plate only", PLATE_ONLY), ("3-2-1 datum", DATUM_321)):
    s = np.linalg.svd(rows(touches), compute_uv=False)
    rank = int(np.sum(s > 1e-9))
    print(f"{name}: rank {rank}, singular values {np.round(s, 3)}")

assert np.linalg.matrix_rank(rows(PLATE_ONLY)) == 3
assert np.linalg.matrix_rank(rows(DATUM_321)) == 6
