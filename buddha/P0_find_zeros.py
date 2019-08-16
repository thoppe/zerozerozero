import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import pixelhouse as ph

alpha = 2.0
extent = 1.5
resolution = 400

N = 500000*10
iterations = 100

def get_iterations(N, alpha, iterations):

    C = np.random.normal(size=(N,2))
    #C = np.random.uniform(-extent, extent, size=(N,2))
    
    C = C[:, 0] + C[:, 1]*1J
    Z = np.zeros_like(C)

    data = []
    for _ in tqdm(range(iterations)):
        Z = Z**alpha + C
        data.append(Z)

    # Drop the points that don't escape
    idx = np.isnan(data[-1])
    data = np.hstack([z[idx] for z in data])

    # Drop the nans
    data = data[~np.isnan(data)]

    X, Y = data.real, data.imag
    return X,Y

def pts_to_bins(X, Y, resolution, dx):
    
    idx = (np.abs(X)>=(extent-dx)) | (np.abs(Y)>=(extent-dx))
    X = X[~idx]; Y = Y[~idx]

    binx = np.round(X/dx).astype(int)
    biny = np.round(Y/dx).astype(int)

    binx += resolution//2
    biny += resolution//2

    img = np.zeros(shape=c.shape[:2])
    for i,j in tqdm(zip(binx, biny)):
        img[i,j] += 1

    return img



c = ph.Canvas(resolution, resolution, extent=extent)
dx = (2*c.extent)/c.width

ITR = [100, 200, 300]
for k, iterations in enumerate(ITR):
    
    X, Y = get_iterations(N, alpha, iterations=iterations)

    img = pts_to_bins(X, Y, resolution, dx)
    img /= img.max()
    img *= 255
    img *= 1.2
    img = img.astype(c.img.dtype)
    c.img[:,:,k] = img


c.show()
