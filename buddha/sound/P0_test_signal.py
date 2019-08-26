import numpy as np
from tqdm import tqdm

from src.S1_compute_spectrum import get_samples
import src.S0_compute_brot as S0

import scipy.io.wavfile as wav
import pixelhouse as ph

f_wav = "source/neptunon - Mille - Crysteena.wav"
samplerate, samples = wav.read(f_wav)

# Convert to mono by dropping channel? (better way I'm sure)
if len(samples.shape)>1:
    samples = samples[:, 0]

fps = 30
f, t, Sxx = get_samples(samplerate, samples, time_cutoff=60,sigma=10)

# Interpolate to a new sample rate
t2 = np.linspace(0, t.max(), t.max()*fps)

Sxx2 = np.zeros(shape=(len(f), len(t2)))
for i, y in enumerate(Sxx[:]):
    yc = np.interp(t2, t, y)
    Sxx2[i] = yc

t = t2; Sxx = Sxx2

Sxx /= Sxx.max()

#import pylab as plt
#plt.pcolormesh(t, f, Sxx)
#plt.xlabel("time [sec]")
#plt.ylabel("freq [Hz]")
#plt.show()


'''
import seaborn as sns
import pylab as plt
sns.distplot(Sxx.sum(axis=0))
plt.show()
print(Sxx.min(), Sxx.max())
exit()
'''

c = ph.Canvas(S0.resolution, S0.resolution, extent=S0.extent)

spectrum = Sxx.T

for i,y in enumerate(tqdm(spectrum)):
    
    color = np.clip(255*(y.sum()/2),0,255).astype(np.uint8)
    
    idx = S0.grid_targets(5, y)
    #idx2 = S0.grid_targets(7, y)
    #idx3 = S0.grid_targets(10, y)

    c = c.blank()
    c.img[idx, :3] = color
    #c.img[idx, 0] = color
    #c.img[idx2, 1] = color
    #c.img[idx3, 2] = color

    c.save(f"data/img/{i:08}.jpg")

    c.show(1)
