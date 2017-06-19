import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
import librosa as librosa
import obspy.signal.filter as osf
import librosa.display
import sys

def findTonicFromChroma(x_chroma):
    n_intense = np.zeros(12)
    for i in range(0, 12):
        # calculate sum of intensities
        n_intense[i] = np.average(x_chroma[i, :])
    C_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                        2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    C_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                        2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    v = -1
    key = -1
    for i in range(0, 12):
        major = np.roll(C_major, i)
        minor = np.roll(C_minor, i)
        corr_major = scipy.stats.pearsonr(major, n_intense)[0]
        corr_minor = scipy.stats.pearsonr(minor, n_intense)[0]
        if (corr_major > v):
            v = corr_major
            key = i
        if (corr_minor > v):
            v = corr_minor
            key = i
    return key


def SW_len(s1, s2):
    n1 = s1.shape[0]
    n2 = s2.shape[0]
    H = np.empty((n1 + 1, n2 + 1))
    for i in range(0, n1 + 1):
        H[i][0] = 0
    for j in range(0, n2 + 1):
        H[0][j] = 0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            simi = np.dot(s1[i - 1], s2[j - 1])
            if (simi > 0.8):
                simi = 4
            else:
                simi = -3
            H[i][j] = max(0, H[i-1][j-1]+simi, H[i-1][j]-1, H[i][j-1]-1)
    return np.amax(H)


# Load audio recordings
x_1, fs = librosa.load(sys.argv[1])
x_1 = osf.bandpass(data=x_1, freqmin=75, freqmax=4000, df=fs, zerophase=True)
x_2, fs = librosa.load(sys.argv[2])
x_2 = osf.bandpass(data=x_2, freqmin=75, freqmax=4000, df=fs, zerophase=True)

# Extract Chroma Features
window_size = 4420  # window size for STFT, 200ms
hop_size = 2210  # step-size for frames, 100ms
x_1_chroma = librosa.feature.chroma_stft(
    y=x_1, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=window_size)
x_2_chroma = librosa.feature.chroma_stft(
    y=x_2, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=window_size)

K1 = findTonicFromChroma(x_1_chroma)
K2 = findTonicFromChroma(x_2_chroma)

feat1 = np.transpose(x_1_chroma)
feat2 = np.transpose(np.roll(x_2_chroma, (K1 - K2) % 12, axis=0))

dist = SW_len(feat1, feat2)
m1 = SW_len(feat1, feat1)
m2 = SW_len(feat2, feat2)

n_dist = 10000*dist/(m1+m2)

s1 = sys.argv[1][0:sys.argv[1].find('(')].split()[0]
s2 = sys.argv[2][0:sys.argv[2].find('(')].split()[0]
c1 = sys.argv[1][sys.argv[1].find('(')+1:sys.argv[1].find(')')].split()[0]
c2 = sys.argv[2][sys.argv[2].find('(')+1:sys.argv[2].find(')')].split()[0]
print s1, '-', s2, ',', n_dist, ',', c1, '-', c2