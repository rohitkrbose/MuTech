import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
import librosa as librosa
import librosa.display
import sys
import obspy.signal.filter as osf
from sklearn.cluster import KMeans
from pyemd import emd

def findTonicFromChroma (x_chroma):
    n_intense = np.zeros(12)
    for i in range (0,12):
        n_intense[i] = np.average(x_chroma[i,:]) # calculate sum of intensities
    C_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    C_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    v = -1
    key = -1
    for i in range (0,12):
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

def retArr (K, feat):
	m = K.cluster_centers_.shape[0]
	Z_mu = np.zeros((m,feat.shape[1]))
	Z_var = np.zeros(m)
	Z_w = np.zeros(m)
	for i in range (0, m):
		X = feat[K.labels_== i]
		centroid = K.cluster_centers_[i]
		mu = np.mean(X, axis=0)
		var = 0
		for j in range (0, X.shape[0]):
			var += np.linalg.norm(X[j]-centroid)
		Z_mu[i] = mu
		Z_var[i] = var
		Z_w[i] = X.shape[0]
	return Z_mu,Z_var,Z_w

def distMat (P_mu, P_var, Q_mu, Q_var, m):
	d = np.zeros((m,m))
	for i in range (0,m):
		for j in range (0,m):
			d[i][j] = P_var[i]/Q_var[j] + Q_var[j]/P_var[i] + scipy.spatial.distance.sqeuclidean(P_mu[i],Q_mu[j])*(1/P_var[i] + 1/Q_var[j])
	return d

# Load audio recordings
x_1, fs = librosa.load(sys.argv[1])
x_1 = osf.bandpass(data=x_1, freqmin=75, freqmax=4000, df=fs, zerophase=True)
x_2, fs = librosa.load(sys.argv[2])
x_2 = osf.bandpass(data=x_2, freqmin=75, freqmax=4000, df=fs, zerophase=True)

# Extract chroma
n_fft = 4420
hop_size = 2210

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)

k1 = findTonicFromChroma(x_1_chroma)
k2 = findTonicFromChroma(x_2_chroma)

feat1 = np.transpose(x_1_chroma)
feat2 = np.transpose(np.roll(x_2_chroma,(k1-k2)%12, axis=0))

m = 7 # number of clusters

K1 = KMeans(n_clusters=m, random_state=0).fit(feat1)
K2 = KMeans(n_clusters=m, random_state=0).fit(feat2)

n1 = feat1.shape[0]
n2 = feat2.shape[0]
P_mu,P_var,P_w = retArr(K1,feat1)
Q_mu,Q_var,Q_w = retArr(K2,feat2)
d = distMat(P_mu, P_var, Q_mu, Q_var, m)

dist = emd(P_w,Q_w,d)

s1 = sys.argv[1][0:sys.argv[1].find('(')].split()[0]
s2 = sys.argv[2][0:sys.argv[2].find('(')].split()[0]
c1 = sys.argv[1][sys.argv[1].find('(')+1:sys.argv[1].find(')')].split()[0]
c2 = sys.argv[2][sys.argv[2].find('(')+1:sys.argv[2].find(')')].split()[0]
print s1, '-', s2, ',', dist, ',', c1, '-', c2