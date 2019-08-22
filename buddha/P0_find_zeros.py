import numpy as np
from tqdm import tqdm
import joblib
import sys, os, glob
import scipy.signal
import h5py
from numba import njit, jit, prange

np.warnings.filterwarnings("ignore")

alpha = 1.5
#alpha = 2.0
extent = 1.5

# 3840x2160p is youtube 4K
# resolution = 1600
resolution = 2048

chunk_size = 50000*10
n_chunks = 40
iterations = 500

save_dest = "data/points"
os.system(f"mkdir -p {save_dest}")

@njit(parallel=True)
def complex_equation(Z, C, alpha):
    return Z ** alpha + C

@njit(parallel=True)
def generate_starting_points(C, N):
    '''
    From a list of starting complex numbers, choose N points
    at some distance sigma randomly from them.
    '''

    # Pick from the targets
    zi = np.random.choice(C, size=(N,))

    # Add some noise, equal to twice the pixel size
    sigma = ((2 * extent) / resolution) * 2
    zi += np.random.normal(0.0, sigma, size=(N,))
    zi += np.random.normal(0.0, sigma, size=(N,)) * 1j

    return zi

def grid_targets(alpha, iterations):

    # Create a complex grid
    line = np.linspace(-extent, extent, resolution)
    grid = np.meshgrid(line, line)
    x, y = grid[0].ravel(), grid[1].ravel()
    C = x + y * 1j

    # Map Mandelbrot over it a few times to find which points
    # are still in the set given a few iterations
    Z = np.zeros_like(C)
    #for _ in tqdm(range(iterations)):
    for _ in tqdm(range(iterations)):
        Z = complex_equation(Z, C, alpha)

    idx = np.isnan(Z) | np.isinf(Z)
    in_set = (~idx).reshape([resolution, resolution])

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

    print(f"Boundary points are {targets.mean()*100:0.2f}% pixels")

    return zi

def get_iterations(N, alpha, iterations, zi):

    C = generate_starting_points(zi, N)
    Z = np.zeros_like(C)

    data = []

    for _ in range(iterations):
        Z = complex_equation(Z, C, alpha)
        data.append(Z)

    # Drop the starting points that didn't escape escape
    idx = np.isnan(data[-1]) | np.isinf(data[-1])
    data = np.array(data)[:, idx]

    # Drop the nans and infinite
    idx = np.isnan(data) | np.isinf(data)
    data = data[~idx]

    data = np.vstack([data.real, data.imag]).T
    return data


def pts_to_bins(pts, resolution, extent):
    rg = [[-extent, extent], [-extent, extent]]
    img, _ = np.histogramdd(pts, bins=(resolution, resolution), range=rg)
    counts = img.astype(np.uint64)
    return counts

def compute_set(chunk_size, alpha, iterations, resolution, extent):
    counts = np.zeros((resolution, resolution), dtype=np.uint64)

    for _ in tqdm(range(n_chunks)):
        pts = get_iterations(chunk_size, alpha, iterations, zi)
        counts += pts_to_bins(pts, resolution, extent)

    return counts



# Construct an empty file with the arguments
f_save = os.path.join(
    save_dest, f"{iterations}_{resolution}_{alpha:0.5f}.h5"
)
if not os.path.exists(f_save):
    with h5py.File(f_save, "w") as h5:
        h5.attrs["resolution"] = resolution
        h5.attrs["iterations"] = iterations
        h5.attrs["extent"] = extent
        h5.attrs["alpha"] = alpha
        h5["counts"] = np.zeros(
            shape=(resolution, resolution), dtype=np.uint32
        )

h5 = h5py.File(f_save, "r+")

if "zi" not in h5:
    print("Finding the boundary")
    zi = grid_targets(alpha, iterations)
    h5["zi"] = zi
else:
    zi = h5["zi"][...]


args = (chunk_size, alpha, iterations, resolution, extent)
counts = compute_set(*args)

print(f"Intermediate {counts.sum()/10**6:0.1f}*10**6 points")

h5["counts"][...] = h5["counts"][...] + counts
counts = h5["counts"][...]
h5.close()

print(f"Final {counts.sum()/10**6:0.1f}*10**6 points")

##########################################################################
