from tqdm import tqdm
import h5py, os
import matplotlib
import numpy as np
from functools import partial
import pixelhouse as ph


circle_radius = 0.0005
pal = ph.palette(0)
bg_color = np.array(pal[0]) / 40

# canvas = ph.Canvas(800, 800, bg=bg_color, extent=2)
canvas = ph.Canvas(1280, 1280, bg=bg_color, extent=2.25)

save_dest = "ph_figures"
os.system(f"mkdir -p {save_dest}")

cutoff = 10 ** 4


def get_frame(k):

    f_save = os.path.join(f"data/point_clouds/", f"points_{k:06d}.h5")
    if not os.path.exists(f_save):
        raise ValueError(f"Cant find {f_save}")

    with h5py.File(f_save, "r") as h5:

        points = h5["points"][:cutoff, ...]

        base_points = h5["base_points"][...]
        t = h5.attrs["t"]

    return points, base_points, t


##################################################################


# Identify the pairs
_, base_points, _ = get_frame(0)


yp = base_points.T[1]
idx = np.argsort(np.argsort(yp))
c0 = np.where(np.isin(idx, [0, 5]))[0]
c1 = np.where(np.isin(idx, [1, 4]))[0]
c2 = np.where(np.isin(idx, [2, 3]))[0]

cidx = np.array([c0[0], c0[1], c1[0], c1[1], c2[0], c2[1]])


for frame_n in range(0, 10000):

    # f_save = f'ph_figures/{frame_n:06d}.png'
    # if os.path.exists(f_save):
    #    print(f"Skipping {f_save}")
    #    continue

    try:
        pt, bp, t = get_frame(frame_n)
    except ValueError:
        break

    bp = bp.T
    pt = pt.T

    """
    BP = np.array([

        [bp[0][cidx[0]], bp[1][cidx[0]]],
        [bp[0][cidx[1]], bp[1][cidx[1]]],
        
        [bp[0][cidx[2]], bp[1][cidx[2]]],
        [bp[0][cidx[3]], bp[1][cidx[3]]],

        [bp[0][cidx[4]], bp[1][cidx[4]]],
        [bp[0][cidx[5]], bp[1][cidx[5]]],

    ])
    """

    COLORS = [pal[1], pal[1], pal[2], pal[2], pal[3], pal[3]]

    COLORS = [pal[1]] * 6

    """
    for _ in range(20):
        canvas += ph.filters.gaussian_blur(0.5, 0.5)
    """

    def draw_color_points(canvas):

        c = np.array(COLORS[0])
        c_dim = c // 10

        img = canvas.img.astype(np.float)
        print(img.sum())

        for q in tqdm(pt.T[:]):

            px = canvas.transform_x(q[0])
            py = canvas.transform_y(q[1])

            if px >= canvas.shape[1]:
                continue

            if py >= canvas.shape[0]:
                continue

            if px < 0 or py < 0:
                continue

            img[px, py] += c_dim
            # img[px, py]  = np.max

        img = np.clip(img, 0, 255).astype(np.uint8)
        canvas.img = img
        print(img.sum())

    with canvas.layer() as C2:
        draw_color_points(C2)
        # C2 += ph.filters.gaussian_blur(0.02, 0.02)
        # draw_color_points(C2)

    # Draw the centroids a light blurry glow
    with canvas.layer() as C2:
        for (x, y), c in zip(bp.T, COLORS):
            C2 += ph.circle(x, y, r=0.025, color=pal[4])
        C2 += ph.filters.gaussian_blur(0.4, 0.4)
        for (x, y), c in zip(bp.T, COLORS):
            C2 += ph.circle(x, y, r=0.025, color=pal[4])

    canvas.show()
    exit()

    output = canvas.copy()
    output.img = output.img[128:-128, :]
    output.save(f_save)
    output.show(1)

    canvas = canvas.blank()
    prior_bp = bp
