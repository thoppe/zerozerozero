import numpy as np
from tqdm import tqdm
import joblib
import sys, os, glob
import scipy.signal
import h5py
import joblib
import tempfile

alpha = 1.5
#alpha = 2.0
extent = 1.5

# 3840x2160p is youtube 4K
# resolution = 1600
resolution = 2048

N = 50000
parallel_iterations = joblib.cpu_count()
inner_loops = 200

save_dest = "data/points"
os.system(f"mkdir -p {save_dest}")

ITERS = [100, 200, 500]


def complex_equation(Z, C, alpha):
    return Z ** alpha + C


def grid_targets(alpha, iterations):
    np.warnings.filterwarnings("ignore")

    # Create a complex grid
    line = np.linspace(-extent, extent, resolution)
    grid = np.meshgrid(line, line)
    x, y = grid[0].ravel(), grid[1].ravel()
    C = x + y * 1j

    # Map Mandelbrot over it a few times to find which points
    # are still in the set given a few iterations
    Z = np.zeros_like(C)
    for _ in range(iterations):
        Z = complex_equation(Z, C, alpha)
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

    # for _ in tqdm(range(iterations)):
    for _ in range(iterations):
        Z = complex_equation(Z, C, alpha)
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


def compute_set(N, alpha, iterations, resolution, extent, seed, save_dest):
    np.warnings.filterwarnings("ignore")
    np.random.seed(seed)

    counts = np.zeros((resolution, resolution), dtype=np.uint64)

    for _ in range(inner_loops): 
        pts = get_iterations(N, alpha, iterations, zi)
        counts += pts_to_bins(pts, resolution, extent)

    f_save = os.path.join(save_dest, f'{seed}.npy')
    np.save(f_save, counts)
    # print(f"Found {counts.sum()/10**6:0.1f}*10**6 points")
    #return None
    # return counts


for i, iterations in enumerate(ITERS):

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

    # Drop the objects into the queue

    func = joblib.delayed(compute_set)
    with joblib.Parallel(-1
    ) as MP, tempfile.TemporaryDirectory() as TMP:
        
        args = (N, alpha, iterations, resolution, extent)
        ITR = tqdm(range(parallel_iterations))

        for res in MP(func(*args, seed=k, save_dest=TMP) for k in ITR):
            pass

        # Accumulate the results
        counts = np.zeros((resolution, resolution), dtype=np.uint64)

        for f_npy in tqdm(glob.glob(os.path.join(TMP, '*'))):
            counts += np.load(f_npy)

    print(f"Intermediate {counts.sum()/10**6:0.1f}*10**6 points")

    h5["counts"][...] = h5["counts"][...] + counts
    counts = h5["counts"][...]
    h5.close()

    print(f"Final {counts.sum()/10**6:0.1f}*10**6 points")

##########################################################################
