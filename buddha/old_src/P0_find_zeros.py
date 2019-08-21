import numpy as np
from tqdm import tqdm
import joblib
import sys, os
import scipy.signal
import pixelhouse as ph
import h5py
import ray
from ray.tune.util import pin_in_object_store, get_pinned_object

#alpha = 1.5
alpha = 2.0

extent = 1.5

# 3840x2160p is youtube 4K 
#resolution = 1600
resolution = 2048

N = 50000
parallel_iterations = 200

save_dest = 'data/points'
os.system(f'mkdir -p {save_dest}')

ray.init()

ITERS = [100,]# 200, 500]


def complex_equation(Z, C, alpha):
    return Z**2 + C

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
        Z = Z**2 + C
        #Z = complex_equation(Z, C, alpha)
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
        Z = Z**2 + C
        #Z = complex_equation(Z, C, alpha)
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



@ray.remote
def compute_set(
        N, alpha, iterations, resolution, extent,
        #zi, seed=None
        zi_obj, seed=None
        
):
    np.warnings.filterwarnings('ignore')
    np.random.seed(seed)

    zi = get_pinned_object(zi_obj)
        
    pts = get_iterations(N, alpha, iterations, zi)
    counts = pts_to_bins(pts, resolution, extent)
    #print(f"Found {counts.sum()/10**6:0.1f}*10**6 points")

    return counts

#iterations = 500

#ITERS = [100, 200, 500]


for i, iterations in enumerate(ITERS):

    # Construct an empty file with the arguments
    f_save = os.path.join(
        save_dest,
        f'{iterations}_{resolution}_{alpha:0.5f}.h5',
    )
    if not os.path.exists(f_save):
        with h5py.File(f_save, 'w') as h5:
            h5.attrs['resolution'] = resolution
            h5.attrs['iterations'] = iterations
            h5.attrs['extent'] = extent
            h5.attrs['alpha'] = alpha
            h5['counts'] = np.zeros(
                shape=(resolution,resolution), dtype=np.uint32)

    h5 = h5py.File(f_save, 'r+')

    if 'zi' not in h5:
        zi = grid_targets(alpha, iterations)
        h5['zi'] = zi
    else:
        zi = h5['zi'][...]

    # Drop the objects into the queue
    object_ids = []

    zi_obj = pin_in_object_store(zi)

    for k in range(parallel_iterations):
        args = (N, alpha, iterations, resolution, extent, zi_obj)
        obj = compute_set.remote(*args, seed=k)
        object_ids.append(obj)

    # Accumulate the results
    counts = np.zeros((resolution, resolution), dtype=np.uint64)


    with tqdm(total=len(object_ids)) as progress:
        while len(object_ids):
            obj, object_ids = ray.wait(object_ids, num_returns=1)
            counts += ray.get(obj[0]).copy()
            progress.update()
        
    print(f"Intermediate {counts.sum()/10**6:0.1f}*10**6 points")

    h5['counts'][...] = h5['counts'][...] + counts
    counts = h5['counts'][...]
    h5.close()
    
    print(f"Final {counts.sum()/10**6:0.1f}*10**6 points")


canvas = ph.Canvas(resolution, resolution, extent=extent)

for i, iterations in enumerate(ITERS):
    f_save = os.path.join(
        save_dest,
        f'{iterations}_{resolution}_{alpha:0.5f}.h5',
    )
    with h5py.File(f_save, 'r') as h5:
        counts = h5['counts'][...]

    img = bins_to_image(counts, resolution, 2)   
    canvas.img[:, :] = img[:, :, np.newaxis]
    #canvas.img[:, :, i] = img[:, :]
    
canvas.resize(0.5).show()
