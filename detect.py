#!/usr/bin/env python

import argparse
import sys

from typing import List
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, lfilter, butter, freqz
from scipy.signal import find_peaks_cwt, find_peaks
from matplotlib.animation import FuncAnimation
import sounddevice as sd

"""Some useful note frequencies."""
A4 = 440.00
C4 = A4*(2**(-9.0/12.0)) # 9 semitones from C4 to A4, so lower 9 semitones to get C4

"""Number of octaves to inspect."""
n_octaves = 4

"""Range of frequencies to observe."""
f_s = 110.0*(2**(-9.0/12.0))
f_e = f_s*(2**n_octaves)


class Note(object):
    """Represents a single note."""

    """Note names."""
    names = {
            0: "C",
            1: "C#/Db",
            2: "D",
            3: "D#/Eb",
            4: "E",
            5: "F",
            6: "F#/Gb",
            7: "G",
            8: "G#/Ab",
            9: "A",
            10: "A#/Bb",
            11: "B"
            }

    def __init__(self, octave: int, semitone: int, off_by: float):
        """
        Constructor.
        :param int octave: In which octave?
        :param int semitone: Which semitone? Between 0 and 12.
        :param float off_by: By how much it is off in fractions of a semitone.
        """
        self.octave = octave
        self.semitone = semitone
        self.off_by = off_by
    def name(self) -> str:
        """Return the conventional name of this note."""
        assert (self.semitone >= 0 and self.semitone < 12), f"The semitone must be in [0, 12). I received {self.semitone}."
        return Note.names[self.semitone]
    def __str__(self) -> str:
        """Return a string representation of this note."""
        name = self.name()
        octave = self.octave
        return f"{name}{octave}, off by {self.off_by:.2f} semitones"
    def __repr__(self) -> str:
        """Python representation of this object."""
        return str(self)

def single_note(f: float) -> Note:
    """
    Convert a frequency into a parsed Note object.
    :param float f: Frquency in Hz.
    return Note: Object with all the note information.
    """
    # f/C4 is 1 if we are exactly at C4
    # f/C0 is 2^4 if we are exactly at C4 and 1 if we are at C0
    # Assume C0 is the lowest possible result

    # if is a power of 2 if we are in a different octave
    # it is multiplied by 2^(n/12.0) if we are n semitones away
    log_f = np.log2(f) - np.log2(C4) + 4

    # in which semitone are we? take only the fractional part and find the closest fraction n/12 to get n
    semitone = log_f - int(log_f)

    # if we are at C(n), then f/C0 = 2^n and log2(f/C0) = n
    octave = int(log_f)

    assert semitone > 0, f"Semitone must be positive, as we subtracted the integer part without rounding. Obtained: semitone={semitone}, from log(f)={log_f}"
    semitone = 12.0*semitone

    closest_semitone = int(round(semitone))
    off_by = semitone - closest_semitone

    # round may round up
    if closest_semitone >= 12:
        closest_semitone -= 12
        octave += 1

    return Note(octave=octave, semitone=closest_semitone, off_by=off_by)

def convert_to_note(freqs: np.ndarray) -> List[Note]:
    """
        Convert an array of float frequencies to a list of note names.
        :param np.ndarray freqs: Frequencies in Hz.
        return List[Note]: List of notes.
    """
    return [single_note(f) for f in freqs]

def remove_harmonics(notes: List[Note]) -> List[Note]:
    """
        Keep all notes that are the same in the input, but correspond to different octaves.
        :param List[Note] notes: Input note list.
        return List[Note]: List keeping only lowest octave.
    """
    my_notes = list()
    for semitone in range(12):
        if any([note.semitone == semitone for note in notes]):
            this_note = [note for note in notes if note.semitone == semitone]
            octaves = [note.octave for note in this_note]
            lowest_octave = np.argmin(np.array(octaves))
            my_notes.append(this_note[lowest_octave])
    return my_notes

def fetch_audio(fs:float,
                N: int,
                play_tone: bool) -> np.ndarray:
    """
        Read the audio and convert it to a Numpy array.
        :param float fs: Sampling frequency in Hz.
        :param int N: Number of samples to take.
        :param bool play_tone: Whether to also play a 440 Hz tone for testing.
        return np.ndarray: Time series of the sound taken.
    """
    if play_tone:
        A = 0.1
        time = N/fs
        x = np.linspace(0, time, N)
        y = A*np.sin(2*np.pi*440.0*x) + A*0.5*np.sin(2*np.pi*880.0*x)
        y = sd.playrec(y, samplerate=fs, channels=2)
    else:
        y = sd.rec(N, samplerate=fs, channels=2)
    sd.wait()
    return y

def live_plotter(i: int, play_tone: bool, min_snr: float):
    """
        Collect audio and plot its Fourier transform.
        :param int i: Frame number used in the plotting.
        :param bool play_tone: Whether to record and play simultaneously to test it.
        :param float min_snr: Minimum signal-to-noise ratio used to filter out noise.
    """
    time = 1.0
    fs = 44.1e3
    Ts = 1.0 / fs
    N = int(time / Ts)

    x = np.linspace(0.0, N*Ts, N)
    y = fetch_audio(fs, N, play_tone=play_tone)
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

    med_bkg = np.median(Y)
    qup_bkg = np.quantile(Y, 0.975)
    qdw_bkg = np.quantile(Y, 0.025)
    std_bkg = 0.25*(qup_bkg - qdw_bkg)
    thr_bkg = med_bkg + min_snr*std_bkg

    plt.cla()
    ax = plt.gca()
    ax.plot(F, Y, '-b', alpha=0.8, label='Fourier transform')
    plt.axhline(y=med_bkg, linestyle='--', lw=2, label='Median', color='green')
    plt.axhline(y=qup_bkg, linestyle='--', lw=2, label='97.5% quantile', color='cyan')
    plt.axhline(y=qdw_bkg, linestyle='--', lw=2, label='2.5% quantile', color='magenta')
    #plt.axhline(y=thr_bkg, linestyle='--', lw=2, label='Threshold', color='red')
    ax.set(xlabel='Frequency [Hz]',
           ylabel='Relative power [dB]',
           title='',
           ylim=(-140.0, 0.0),
           xlim=(f_s, f_e))
    idx, properties = find_peaks(Y, height=thr_bkg)
    #idx = find_peaks_cwt(Y, np.arange(1, 10), min_snr=min_snr)
    if len(idx) > 0:
        ax.scatter(F[idx], Y[idx], s=40, marker='o')
        peaks_freq = F[idx]
        notes = convert_to_note(peaks_freq)
        only_fundamental = remove_harmonics(notes)
        print(f"Notes found: {only_fundamental}")
        print(f"   -> Peaks: {peaks_freq}")
        print(f"   -> All modes found: {notes}")
    plt.legend(frameon=False)

def show_filter_response():
    """Make a plot of the filter response."""
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Detect the frequency of a tone.')
    parser.add_argument('--show-filter-response',
                        action='store_true',
                        help='Show the filter response?')
    parser.add_argument('--play-tone',
                        action='store_true',
                        help='Play a 440 Hz tone to calibrate it.')
    parser.add_argument('--min-snr',
                        type=float,
                        metavar='FLOAT',
                        action='store',
                        default=3.0,
                        help='Minimum signal-to-noise ratio used to filter ambient noise.')

    args = parser.parse_args()
    print(f"Minimum signal-to-noise ratio: {args.min_snr:.2f}")

    if args.show_filter_response:
        show_filter_response()
        sys.exit(0)

    #plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(10, 8))
    ani = FuncAnimation(fig, live_plotter, interval=100, fargs=(args.play_tone, args.min_snr))

    #plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    main()

