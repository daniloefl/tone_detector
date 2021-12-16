#!/usr/bin/env python

import re
import argparse
import os
import sys

from typing import List

import numpy as np
import sounddevice as sd

"""Some useful note frequencies."""
A4 = 440.00
C4 = A4*(2**(-9.0/12.0)) # 9 semitones from C4 to A4, so lower 9 semitones to get C4
C0 = C4*(2**(-4))

class Note(object):
    """Represents a single note."""

    """Note names."""
    names = {
            0: "C",
            1: "C#",
            2: "D",
            3: "D#",
            4: "E",
            5: "F",
            6: "F#",
            7: "G",
            8: "G#",
            9: "A",
            10: "A#",
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
    def f(self) -> float:
        return C0*(2**(self.octave + ((self.semitone + self.off_by)/12.0)))

def name_to_note(name: str) -> Note:
    """
        Convert a name to the note object.
    """
    res = re.search(r"([A-G]?\#?)(\d+)([-+][0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)?", name)
    note_name = res.groups()[0]
    semitone = -1
    for s, expected in Note.names.items():
        if expected == note_name:
            semitone = s
    assert semitone > -1
    octave = float(res.groups()[1])
    off_by = 0.0
    if res.groups()[2] is not None:
        off_by = float(res.groups()[2])
    return Note(octave, semitone, off_by)

def play_notes(f: List[float],
              fs:float,
              N: int,
              ):
    """
        Play notes.
        :param List[float] f: Frquencies to play.
        :param float fs: Sampling frequency in Hz.
        :param int N: Number of samples to take.
    """
    A = 0.1
    time = N/fs
    x = np.linspace(0, time, N)
    y = np.zeros_like(x)
    for freq in f:
        y += A*np.sin(2*np.pi*freq*x)
    y = sd.playrec(y, samplerate=fs, channels=2)
    sd.wait()

def play(notes: List[Note]):
    """
        Play tones in list.
        :param List[Note] notes: Notes to play.
    """
    print("Notes to play:", notes)
    time = 10.0 # seconds
    fs = 44.1e3 # Hz
    Ts = 1.0 / fs # seconds
    N = int(time / Ts)

    freqs = [note.f() for note in notes]
    play_notes(freqs, fs, N=N)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Play tones.')
    parser.add_argument('notes', type=str, nargs='+',
                    help='Notes to play.')

    args = parser.parse_args()

    notes = args.notes
    play([name_to_note(note) for note in notes])

if __name__ == '__main__':
    main()

