#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, lfilter, butter, freqz
from scipy.signal import find_peaks_cwt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

def fetch_audio(fs:float,
                N: int) -> np.ndarray:
    y = sd.rec(N, samplerate=fs, channels=2)
    sd.wait()
    return y

def live_plotter(i: int) -> None:
    n_octaves = 4
    time = 1.0
    fs = 44.1e3
    Ts = 1.0 / fs
    N = int(time / Ts)

    f_s = 220.0
    f_e = f_s*n_octaves

    x = np.linspace(0.0, N*Ts, N)
    y = fetch_audio(fs, N)
    y = np.mean(y, axis=-1)

    b, a = butter(4, (f_s, f_e), btype='bandpass', output='ba', analog=False, fs=fs)
    y = lfilter(b, a, y)

    F, Y = welch(y,
                 fs=fs,
                 window="hamming",
                 nperseg=N,
                 )
                 #scaling='spectrum')
    # Y = 1/T |FT(y)[x]|**2 in intensity^2 / Hz
    Y = 10*np.log10(Y) # in dB

    ## filter it
    #window = np.hanning(N)
    #y = y*window

    #Y = np.fft.fft(y)
    #Y = Y[:N//2]

    #Y = 20.0*np.log10(np.abs(Y)) # in dB

    #F = np.fft.fftfreq(x.shape[0], d=Ts)
    #F = F[:N//2]
    ##F = np.linspace(0.0, 0.5*fs, N//2)

    i_s = np.where(F >= f_s)[0][0]
    i_e = np.where(F >= f_e)[0][0]
    select = slice(i_s, i_e)
    F = F[select]
    Y = Y[select]

    plt.cla()
    ax = plt.gca()
    ax.plot(F, Y, '-b', alpha=0.8)
    ax.set(xlabel='Frequency [Hz]',
           ylabel='Relative power [dB]',
           title='',
           ylim=(-140.0, 0.0),
           xlim=(f_s, f_e))
    idx = find_peaks_cwt(Y, np.arange(1, 10))
    if len(idx) > 0:
        print(f"Peaks found in frequencies: {F[idx]}")

def main() -> None:
    #plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    f_s = 220.0
    f_e = 220.0*4
    fs = 44.1e3
    b, a = butter(4, (f_s, f_e), btype='bandpass', output='ba', analog=False, fs=fs)
    w, h = freqz(b, a, fs=fs)
    ax.semilogx(w, 20 * np.log10(abs(h)))
    ax.set(title='Filter response', xlabel='Frequency [Hz]', ylabel='Amplitude [dB]', ylim=(-40, 0), xlim=(f_s/4, 4*f_e))
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(f_s, color='green')
    plt.axvline(f_e, color='green')
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ani = FuncAnimation(fig, live_plotter, interval=100)

    #plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

