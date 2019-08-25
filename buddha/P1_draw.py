import pixelhouse as ph
import os
import numpy as np
import h5py

ITERS = [100, 200, 500]
ITERS = [100]
alpha = 2.0
extent = 1.5
resolution = 2048 // 8
save_dest = "data/points"


def bins_to_image(counts, resolution, boost=1.0):

    _, bins = np.histogram(counts.ravel(), bins=255)
    norm_color = np.digitize(counts, bins, True)

    norm_color = (counts / counts.max()) * 255

    img = np.clip(norm_color * boost, 0, 255).astype(np.uint8)

    return img


canvas = ph.Canvas(resolution, resolution, extent=extent)

for i, iterations in enumerate(ITERS):
    f_save = os.path.join(save_dest, f"{iterations}_{resolution}_{alpha:0.5f}.h5")

    with h5py.File(f_save, "r") as h5:
        counts = h5["counts"][...]
        img = bins_to_image(counts, resolution, 3)

    if len(ITERS) == 1:
        canvas.img[:, :] = img[:, :, np.newaxis]
    else:
        canvas.img[:, :, i] = img[:, :]

# canvas.resize(0.25).show()
canvas.show()
