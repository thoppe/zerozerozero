import pixelhouse as ph
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def get_samples(
        samplerate, samples,
        top_freq = 440*2,
        bottom_freq = 440/2,
):

    f, t, Sxx = signal.spectrogram(samples, fs=samplerate, nperseg=256*2)

    # Only keep a discrete range of freqs
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

    return f, t, Sxx



if __name__ == "__main__":
    f_wav = "source/sample2.wav"
    f_wav = "source/neptunon - Mille - Crysteena.wav"
    
    samplerate, samples = wav.read(f_wav)

    # Convert to mono by dropping channel? (better way I'm sure)
    if len(samples.shape)>1:
        samples = samples[:, 0]

    f, t, Sxx = get_samples(samplerate, samples)


    import pylab as plt
    plt.pcolormesh(t, f, Sxx)
    plt.xlabel("time [sec]")
    plt.ylabel("freq [Hz]")
    plt.show()


