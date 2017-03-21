import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa as librosa
import librosa.display
import sys

# Usage syntax:
# python dtw_chroma.py <song_1> <song_2>

# Load audio recordings
x_1, fs = librosa.load(sys.argv[1], offset=30, duration=60)
x_2, fs = librosa.load(sys.argv[2], offset=30, duration=60)

# Extract Chroma Features
n_fft = 4096 # window size for STFT
hop_size = 2048 # step-size for frames

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft) # feature vectors
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft) # feature vectors

plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.title('Chroma Representation of ' + sys.argv[1])
librosa.display.specshow(x_1_chroma, x_axis='time', y_axis='chroma', cmap='magma_r', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Chroma Representation of ' + sys.argv[2])
librosa.display.specshow(x_2_chroma, x_axis='time', y_axis='chroma', cmap='magma_r', hop_length=hop_size)
plt.colorbar()
plt.tight_layout()

# Align Chroma Sequences
D, wp = librosa.core.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine') # D: cumulative cost matrix, wp: warp path indices
print(D[D.shape[0]-1][D.shape[1]-1]) # last co-ordinate has the total alignment cost

wp_s = np.asarray(wp) * hop_size / fs

# Plot Cost Matrix
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
librosa.display.specshow(D, x_axis='time', y_axis='time', cmap='viridis_r', hop_length=hop_size)
imax = ax.imshow(D, cmap=plt.get_cmap('viridis_r'), origin='lower', interpolation='nearest', aspect='auto')
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
plt.title('Warping Path on Cumulative Cost Matrix')
plt.colorbar()

plt.show()
