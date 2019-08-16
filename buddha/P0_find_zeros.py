import mpmath as mp
import numpy as np
import random
from tqdm import tqdm
import joblib
import sys, os
import pixelhouse as ph

alpha = 2.0
extent = 2
resolution = 800

N = 500000
iterations = 100

def get_iterations(N, alpha):

    C = np.random.normal(size=(N,2))
    #C = np.random.uniform(-extent, extent, size=(N,2))
    
    C = C[:, 0] + C[:, 1]*1J
    Z = np.zeros_like(C)

    data = []
    for _ in tqdm(range(iterations)):
        Z = Z**alpha + C
        #idx = np.isnan(Z)
        #C[idx] = 0
        #Z[idx] = 0
        data.append(Z)

    idx = np.isnan(data[-1])
    #C = C[idx]
    data = np.hstack([z[idx] for z in data])

    # Drop the nans
    data = data[~np.isnan(data)]

    X, Y = data.real, data.imag
    return X,Y


#X, Y = get_iterations(N, alpha)
#print(X)
#exit()

c = ph.Canvas(resolution, resolution, extent=extent)
dx = (2*c.extent)/c.width

X, Y = get_iterations(N, alpha)

idx = (np.abs(X)>=extent) | (np.abs(Y)>=extent)
X = X[~idx]; Y = Y[~idx]

binx = np.round(X/dx).astype(int)
biny = np.round(Y/dx).astype(int)

binx += resolution//2
biny += resolution//2

idx = (binx < 0) | (biny < 0) | (binx >= resolution) | (biny >= resolution)
binx = binx[~idx]
biny = biny[~idx]

img = np.zeros(shape=c.shape[:2])
for i,j in tqdm(zip(binx, biny)):
    img[i,j] += 1

    '''
    try:
        
    except IndexError:
        print(i,j)
        continue
    '''
    
img /= img.max()
img *= 255

#img = np.log(img+1)
#img /= img.max()
#img *= 255

#img = (img>0)*255

img = img.astype(c.img.dtype)
c.img[:,:,0] = img
c.show()
