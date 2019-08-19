import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import scipy.signal
import pixelhouse as ph

alpha = 2.0
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

    X, Y = data.real, data.imag
    return X, Y


def pts_to_bins(X, Y, resolution, dx):

    idx = (np.abs(X) >= (extent - dx)) | (np.abs(Y) >= (extent - dx))

    print((~idx).mean())

    X = X[~idx]
    Y = Y[~idx]

    binx = np.round(X / dx).astype(int)
    biny = np.round(Y / dx).astype(int)

    binx += resolution // 2
    biny += resolution // 2

    img = np.zeros(shape=c.shape[:2])
    for i, j in tqdm(zip(binx, biny)):
        img[i, j] += 1

    return img


c = ph.Canvas(resolution, resolution, extent=extent)
dx = (2 * c.extent) / c.width

# ITR = [100, 200, 300]
ITR = [100]
for k, iterations in enumerate(ITR):

    X, Y = get_iterations(N, alpha, iterations=iterations)

    img = pts_to_bins(X, Y, resolution, dx)
    img /= img.max()
    img *= 255
    img *= 1.2
    img = img.astype(c.img.dtype)
    c.img[:, :, k] = img


c.show()
