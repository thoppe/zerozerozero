import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import scipy.signal
import pixelhouse as ph

#alpha = 1.5
alpha = 1.2

extent = 1.5
resolution = 800

N = 50000 // 1
parallel_iterations = 50

def grid_targets(alpha, iterations):
    np.warnings.filterwarnings('ignore')
    
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
    zi = C[targets]

    return zi

def generate_starting_points(C):

    # Pick from the targets
    zi = np.random.choice(C, size=(N,))

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


def get_iterations(N, alpha, iterations, zi):

    C = generate_starting_points(zi)
    Z = np.zeros_like(C)

    data = []
    
    #for _ in tqdm(range(iterations)):
    for _ in range(iterations):
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
    counts = img.astype(np.uint64)
    return counts


def bins_to_image(counts, resolution, boost=1.0):

    #_, bins = np.histogram(counts.ravel(), bins=255)
    #norm_color = np.digitize(counts, bins, True)

    norm_color = (counts / counts.max())*255
    img = np.clip(norm_color * boost, 0, 255).astype(np.uint8)

    return img



import ray
ray.init()

@ray.remote
def compute_set(
        N, alpha, iterations, resolution, extent,
        zi, seed=None
):
    np.warnings.filterwarnings('ignore')
    np.random.seed(seed)
        
    pts = get_iterations(N, alpha, iterations, zi)
    counts = pts_to_bins(pts, resolution, extent)
    print(f"Found {counts.sum()/10**6:0.1f}*10**6 points")

    return counts

#iterations = 500

object_ids = []
canvas = ph.Canvas(resolution, resolution, extent=extent)

for i, iterations in enumerate([100, 200, 500]):

    zi = grid_targets(alpha, iterations)

    # Drop the objects into the queue
    for k in range(parallel_iterations):
        args = (N, alpha, iterations, resolution, extent, zi)
        obj = compute_set.remote(*args, seed=k)
        object_ids.append(obj)

    # Accumulate the results
    counts = np.zeros((resolution, resolution), dtype=np.uint64)
    for obj in object_ids:
        counts += ray.get(obj)

    print(f"Final {counts.sum()/10**6:0.1f}*10**6 points")

    img = bins_to_image(counts, resolution)
    
    #canvas.img[:, :] = img[:, :, np.newaxis]
    canvas.img[:, :, i] = img[:, :]
    
canvas.show()

'''
ITR = [100, 200, 500]
ITR = [100]
for k, iterations in enumerate(ITR):

    img = computer_set(N, alpha, iterations, resolution, extent)
    #c.img[:, :, k] = img
    c.img[:, :] = img[:, :, np.newaxis]
    #c.img[:, :, k] = img
'''

