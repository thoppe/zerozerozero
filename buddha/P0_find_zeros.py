import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import scipy.signal
import pixelhouse as ph

alpha = 1.5
extent = 1.5
resolution = 400

N = 500000 // 1
iterations = 100


def generate_starting_points(alpha, iterations):

    # Create a complex grid
    line = np.linspace(-extent, extent, resolution)
    grid = np.meshgrid(line, line)
    x, y = grid[0].ravel(), grid[1].ravel()
    C = x + y * 1j

    # Map Mandelbrot over it a few times to find which points
    # are still in the set given a few iterations
    Z = np.zeros_like(C)
    for _ in range(iterations):
        Z = Z ** alpha + C
    in_set = (~np.isnan(Z)).reshape([resolution, resolution])

    # Uncomment for a quick viz
    # c = ph.Canvas(resolution, resolution, extent=extent)
    # c.img[in_set,0] = 255
    # c.img[targets,1] = 255
    # c.show()
    # exit()

    # Count the number of times it shows up
    kernel = np.ones((3, 3))
    counts = scipy.signal.convolve2d(in_set, kernel, mode="same")

    # Keep only the points on the boundry (>0,<9)
    targets = ((counts > 0) & (counts < 9)).ravel()

    # Pick from the targets
    i, = np.where(targets)
    n_targets = len(i)
    idx = np.random.choice(n_targets, size=(N,))
    zi = C[i[idx]]

    # Add some noise, equal to twice the pixel size
    sigma = ((2 * extent) / resolution) * 2
    zi += np.random.normal(0.0, sigma, size=(N,))
    zi += np.random.normal(0.0, sigma, size=(N,)) * 1j

    # Uncomment for a quick viz
    # c = ph.Canvas(resolution, resolution, extent=extent)
    # for pt in zi:
    #    c += ph.circle(pt.real, pt.imag, r=0.001)
    # c.show()

    return zi

    C = np.random.normal(size=(N, 2))
    C = C[:, 0] + C[:, 1] * 1j

    return C


def get_iterations(N, alpha, iterations):

    C = generate_starting_points(alpha, iterations)
    Z = np.zeros_like(C)

    data = []
    for _ in tqdm(range(iterations)):
        Z = Z ** alpha + C
        data.append(Z)

    # Drop the points that don't escape
    idx = np.isnan(data[-1])
    data = np.hstack([z[idx] for z in data])

    # Drop the nans
    data = data[~np.isnan(data)]

    data = np.vstack([data.real, data.imag]).T
    return data


def pts_to_bins(pts, resolution, extent):
    rg = [[-extent, extent], [-extent, extent]]
    img, _ = np.histogramdd(pts, bins=(resolution, resolution), range=rg)
    return img


def bins_to_image(counts, resolution, boost=1.0):

    _, bins = np.histogram(counts.ravel(), bins=255)
    norm_color = np.digitize(counts, bins, True)

    img = np.clip(norm_color * boost, 0, 255).astype(np.uint8)

    return img


c = ph.Canvas(resolution, resolution, extent=extent)

ITR = [100, 200, 500]
#ITR = [100]
for k, iterations in enumerate(ITR):

    pts = get_iterations(N, alpha, iterations=iterations)
    print(f"Found {len(pts)//10**6}*10**6 points")

    counts = pts_to_bins(pts, resolution, extent)
    img = bins_to_image(counts, resolution, 2.0)

    c.img[:, :, k] = img
    #c.img[:, :, 1] = img
    #c.img[:, :, 2] = img

c.show()
