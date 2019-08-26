import pixelhouse as ph
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage import gaussian_filter1d

f_wav = "source/sample2.wav"
f_wav = "source/neptunon - Mille - Crysteena.wav"

samplerate, samples = wav.read(f_wav)
if len(samples.shape)>1:
    samples = samples[:, 0]
print(samples)

window = ('tukey', 0.25)
f,t,Sxx = signal.spectrogram(
    samples, fs=samplerate, window=window,
    nperseg=256*2
)

# Only keep a discrete range of freqs
bottom_freq = 440/2
top_freq = 1100

idx = (f<bottom_freq)
bot = Sxx[idx].sum(axis=0)

idx = (f>top_freq)
top = Sxx[idx].sum(axis=0)
idx = (f<top_freq) & (f>bottom_freq)

# Roll this missing freqs into range
f = f[idx]
Sxx = Sxx[idx]
Sxx[0,:] += bot
Sxx[-1,:] += top

idx = t<30
t = t[idx]
Sxx = Sxx[:,idx]



# Smooth the signal
sigma = 2*samplerate/10000

for i, y in enumerate(Sxx[:]):
    yc = gaussian_filter1d(y, sigma=sigma)
    Sxx[i] = yc
    #plt.plot(t,y)
    #plt.plot(t,yc)



import pylab as plt
plt.pcolormesh(t, f, Sxx)
plt.xlabel("time [sec]")
plt.ylabel("freq [Hz]")
plt.show()



exit()


binsize = 2 ** 10
sf = stft(samples, binsize)
sshow, freq = logscale_spec(sf, factor=1.0, sr=samplerate)
intensity = np.log10(np.abs(sshow)/10.e-6)*20

print(intensity.max(), intensity.min())

import pylab as plt
plt.imshow(intensity.T, origin="lower", aspect="auto")
plt.xlabel("time (s)")
plt.ylabel("frequency (hz)")
#plt.xlim([0, timebins-1])
#plt.ylim([0, freqbins])
plt.show()

exit()


intensity = np.sqrt(sf.real**2 + sf.imag**2)
#print(sf.real)
print(intensity)
