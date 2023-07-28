import torch
import numpy as np
from torch import linspace
from tqdm import tqdm
import fast_histogram

extent = 3.0
real_min = -1.75
imag_min = -1.5

real_max = real_min + extent
imag_max = imag_min + extent

starting_iterations = 100
counting_iterations = 100

resolution = 256 * 4
N = 1_000_000_000 // 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

scale_grid = 1
grid_size = (resolution * scale_grid, resolution * scale_grid)
dtype = torch.complex128
torch.set_grad_enabled(False)

# Make sure we are using a square image
assert (real_max - real_min) == (imag_max - imag_min)


def iterating_function(Z, C, a2=1.0, a1=0.0, a0=1.0):
    return a2 * Z ** 2 + a1 * Z + a0 * C


def is_point_escaped(ZX):
    escape_magnitude = 2
    idx0 = torch.isnan(ZX)
    idx1 = torch.isinf(ZX)
    idx2 = torch.abs(ZX) > escape_magnitude
    return idx0 | idx1 | idx2


# Build a grid on the complex plane, we start looking here
real_grid = linspace(real_min, real_max, grid_size[0]).view(-1, 1)
imag_grid = linspace(imag_min, imag_max, grid_size[1]).view(1, -1)

C = torch.complex(real_grid, imag_grid)
C = C.reshape([resolution * resolution * scale_grid ** 2]).to(device)

# Determine which starting points are near the Mandelbrot set
Z = torch.zeros_like(C)

escape_time = torch.zeros_like(C, dtype=torch.int32)
for n in tqdm(range(starting_iterations)):
    Z = iterating_function(Z, C)
    idx = (is_point_escaped(Z)) & (escape_time == 0)
    escape_time[idx] = n

idx = escape_time == 0

# Running using CPU gets 800% CPU and ~  6it/sec on 8192 resolution
# Running using GPU gets 100% GPU and ~323it/sec on 8192 resolution
# print(idx.cpu().numpy().astype(int).mean())

# Keep only the points that have survived the starting iterations
# Use a 3x3 kernel and convolution to count any places where we've stayed
# in the Mandelbrot set and one pixel away
in_set = idx.reshape([1, 1, resolution * scale_grid, resolution * scale_grid])
in_set = in_set.to(dtype=torch.float32)

# Larger kernel size allows you to see more of the original fractal
kn_size = 3

kernel = torch.ones((kn_size, kn_size)).view(1, 1, kn_size, kn_size).to(device)
counts = torch.nn.functional.conv2d(in_set, kernel, padding="same")
idx = (counts > 0).ravel()

pts_starting = C[idx]
print(f"Using {idx.cpu().numpy().mean()*100:0.2f}% of starting points")

# Choose some starting points and add some noise equal to twice pixel size
selection_idx = np.random.choice(len(pts_starting), N)
Ci = pts_starting[selection_idx]
sigma = ((2 * (real_max - real_min)) / resolution) * 2

Ci += torch.complex(
    torch.randn(N, device=device) * sigma,
    torch.randn(N, device=device) * sigma,
)


# Run a quick iteration, see which ones escape
Z = torch.zeros_like(Ci)
for n in range(counting_iterations):
    Z = iterating_function(Z, Ci)

# Keep the ones that do escape
idx = is_point_escaped(Z)
Ci = Ci[idx]

idxc = idx.cpu().numpy()
print(f"Found {idxc.sum()} {idxc.mean()} escaping points")


# Run a smaller iteration of good points and build the histogram
Z = torch.zeros_like(Ci)

kwargs = {"range": [[real_min, real_max], [imag_min, imag_max]], "bins": resolution}
imgs = np.zeros([counting_iterations, resolution, resolution])

for n in tqdm(range(counting_iterations)):
    Z = iterating_function(Z, Ci)

    # Only keep points that have not escaped
    idx = ~is_point_escaped(Z)
    Z = Z[idx]
    Ci = Ci[idx]

    # Count the number of points in pixel space
    Z_np = Z.cpu().numpy()
    pts = fast_histogram.histogram2d(Z_np.real, Z_np.imag, **kwargs)
    imgs[n] += pts

print(f"Recorded {int(imgs.sum()):,d} points")

# Flatten the image
img = imgs.sum(axis=0)

# img = img.astype(float)
# img /= img[img > 0].mean() * 10
# img = np.clip(img, 0, 255)
# img += 5
# img = np.log(img)
# img /= img.max()
# img *= 255

img = np.clip(img, 0, 255)
img = img.astype(np.uint8)

import pixelhouse as ph

canvas = ph.Canvas(resolution, resolution, extent=(real_min - real_max))
img = img[:, :, np.newaxis]
canvas.img[:, :] = img
print(canvas)
canvas.show()
