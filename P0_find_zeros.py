import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import h5py
import scipy.stats

save_dest = "data/point_clouds"
os.system(f"mkdir -p {save_dest}")

fps = 60
n_bases = 6

N = 7
n_samples = 25000 * 4

np_sample_seed = 26


def interpolate(n, frames_per):
    T = np.linspace(0, n - 1, n * frames_per)

    low_s = 0.05
    high_s = 1.00

    sigma = np.random.uniform(low=low_s, high=high_s, size=[n])
    G = [scipy.stats.norm(loc=k, scale=s) for k, s in zip(range(n), sigma)]

    ZX = []
    for frame_idx, t in tqdm(enumerate(T), total=len(T)):
        weights = np.array([g.pdf(t) for g in G])

        # Need to wrap around weights for perfect loop
        weights += np.array([g.pdf(t + T.max()) for g in G])
        weights += np.array([g.pdf(t - T.max()) for g in G])

        # Set the strength of the weights to be unity
        weights /= weights.sum()

        ZX.append(weights)
    return ZX


def random_coeffs(N):
    return np.random.normal(size=(N,))


def get_roots(real=None, imag=None):

    if real is None:
        real = random_coeffs(N)
    if imag is None:
        imag = random_coeffs(N)

    coeffs = real + imag * 1j

    # p = mp.polyroots(coeffs, maxsteps=100, extraprec=110)
    p = mp.polyroots(coeffs, maxsteps=50, extraprec=20)
    p = [[float(z.real), float(z.imag)] for z in p]
    return np.array(p)


def sample(seed=None):
    r = None

    if seed is not None:
        np.random.seed(seed)

    while r is None:
        try:
            r = get_roots(real_coeffs)
        except ZeroDivisionError:
            print("Warning, ZeroDivision Error")
            continue

    return r


func = joblib.delayed(sample)


def compute_set(real_coeffs):
    np.random.seed(np_sample_seed)

    roots = []

    with joblib.Parallel(-1) as MP:

        ITR = np.random.randint(0, np.iinfo(np.int16).max, size=(n_samples,))
        for res in MP(func(seed=x) for x in tqdm(ITR)):
            roots.extend(res)

    return np.array(roots)


C_BASE = [random_coeffs(N) for _ in range(n_bases)]
C_BASE.append(C_BASE[0])
C_BASE = np.array(C_BASE)


points = []
base_points = []
T = []

ITR = enumerate(zip(C_BASE, C_BASE[1:]))
weights = interpolate(len(C_BASE), fps)

ITR = zip(np.linspace(0, 1, len(C_BASE) * fps), weights)


frame_n = 0

for t, w in tqdm(ITR, total=len(weights)):

    f_save = os.path.join(save_dest, f"points_{frame_n:06d}.h5")
    if os.path.exists(f_save):

        frame_n += 1
        continue

    real_coeffs = (C_BASE.T * w).sum(axis=1)

    T.append(t)

    base = get_roots(real_coeffs, np.zeros_like(real_coeffs))
    # real, imag = base.T
    # base_points.append([real, imag])

    roots = compute_set(real_coeffs)
    # real, imag = roots.T
    # points.append([real, imag])

    with h5py.File(f_save, "w") as h5:

        h5["poly_real"] = real_coeffs
        h5["poly_weights"] = w
        h5["C_BASE"] = C_BASE
        h5.attrs["t"] = t

        h5["points"] = roots
        h5["base_points"] = base

    print(f_save)

    frame_n += 1

"""
#plt.scatter(real, imag, s=10, alpha=0.75, color='r')
plt.scatter(real, imag, s=1, alpha=0.25)

plt.axis('tight')
sns.despine(left=True, bottom=True)

plt.xlim(-2,2)
plt.ylim(-2,2)

plt.savefig(f'figures/{k:03d}.png')

plt.clf()
plt.cla()
"""

# plt.show()


"""
import pixelhouse as ph
cx = ph.Canvas(800, 800)

for r in tqdm(roots):

    cx += ph.circle(
        r.real, r.imag, r=0.05, color=[25,]*3, mode='add')
    
    cx += ph.filters.gaussian_blur(blur_x=0.01, blur_y=0.01)
cx.show()
"""
