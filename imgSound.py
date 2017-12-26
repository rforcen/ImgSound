'''
image 2 sound conversion
'''
from math import pi
from os import system

import numpy as np
from analytic_wfm import peakdetect
from scipy.io.wavfile import write
from scipy.misc import imread
from scipy.signal import filtfilt, butter

from musicFreq import MusicFreq

'''
generate wav file w/image peak histogram
'''


def imgSound(fnme, wfn, secs=2, sampleRate=44100):
    def fromjpg2int32(fnme):  # convert jpg 2 int color array
        img = imread(fnme)
        return np.frombuffer(
            np.insert(img, 3, values=0, axis=2).
                reshape(img.shape[0] * img.shape[1] * 4).tobytes(), 'I')

    def genHistoPeaks(colors):
        hist = np.histogram(colors, 1024)  # peak histograms
        fh = filtfilt(*butter(3, 0.1), hist[0])
        pk = peakdetect(fh, lookahead=20)
        return pk

    def genWave(amps, freqs, sampleRate, secs):  # amp[i] * sin( freq[i] * t )
        def scale(input, min=0, max=1):
            input -= np.min(input)
            input /= np.max(input) / (max - min)
            input += min
            return input

        inc = 2 * pi / sampleRate
        w = np.mean([a * np.sin(h * np.arange(0, secs * sampleRate * inc, inc)) for a, h in zip(amps, freqs)], axis=0)
        return np.asarray(scale(w, -0x7fff, +0x7fff), dtype=np.int16)  # scale 16 bit

    def genAmpFreq(pk):
        freqs = np.array([MusicFreq.freq2octave(f[0], 0) for f in pk[0]])  # freqs (oct0) & amps
        amps = np.array([f[0] for f in pk[1]])
        return amps, freqs

    def writeWave(fn, wave, sr):
        write(fn, sr, wave)

    colors = fromjpg2int32(fnme)  # generate colors
    pk = genHistoPeaks(colors)  # histograms peaks
    amps, freqs = genAmpFreq(pk)

    wave = genWave(amps, freqs, sampleRate, secs)
    writeWave(wfn, wave, sampleRate)

    # play wav file
    system('afplay ' + wfn)


if __name__ == '__main__':
    imgSound('img0005.jpg', 'wave01.wav') # test w/jpg file
