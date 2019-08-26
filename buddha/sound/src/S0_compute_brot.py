import numpy as np
from numba import njit

np.warnings.filterwarnings("ignore")

# 3840x2160p is youtube 4K
resolution = 2048 // 8
#extent = 1.5
extent = 2.5


# Multiplying by k*Z**2 gives a rotation
# adding a constant removes parts of it {-1,1} seems reasonable
# Good extra factors on their own exp([-1,1]*Z**2), cos, sin, cosh, sinh, sinc

n_terms = 7

@njit(parallel=True)
def complex_equation(Z, C, coeffs):

    Z2 = Z**2

    x = C
    x += coeffs[0] * Z2
    x += coeffs[1] * np.sinc(Z2)
    x += coeffs[2] * np.cosh(Z2)
    
    x += coeffs[3] * np.cos(Z)
    x += coeffs[4] * np.sin(Z)
    x += coeffs[5] * np.cos(Z2)
    x += coeffs[6] * np.sin(Z2)
    x += coeffs[7] * np.exp(Z2)
    
    #x += coeffs[7] * np.exp(coeffs[8]*Z)


    return x


def grid_targets(iterations, coeffs, ):

    #coeffs = np.random.uniform(-1, 1, size=(n_terms,))

    # Create a complex grid
    line = np.linspace(-extent, extent, resolution)
    grid = np.meshgrid(line, line)
    x, y = grid[0].ravel(), grid[1].ravel()
    C = x + y * 1j

    # Map Mandelbrot over it a few times to find which points
    # are still in the set given a few iterations
    Z = np.zeros_like(C)

    coeffs = np.array(coeffs)
    #coeffs /= np.linalg.norm(coeffs)

    for _ in range(iterations):
        Z = complex_equation(Z, C, coeffs)

    idx = np.isnan(Z) | np.isinf(Z)
    in_set = (~idx).reshape([resolution, resolution])

    return in_set



if __name__ == "__main__":


    import pixelhouse as ph

    # Cycle through a bunch of canvas
    c = ph.Canvas(resolution, resolution, extent=extent)
    while True:
        c = c.blank()
        c.img[grid_targets(5), 0] = 255
        c.img[grid_targets(7), 1] = 255
        c.img[grid_targets(10), 2] = 255
        c.show(1)
