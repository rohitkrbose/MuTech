import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as lrdisp
import sys
from math import *
from mutagen.mp3 import MP3

def f(x):
	e = 0.05
	return (x+e)/(144+24*e)

def stateSpace ():
	return ["C:maj","C#:maj","D:maj","Eb:maj","E:maj","F:maj","F#:maj","G:maj","Ab:maj","A:maj","Bb:maj","B:maj","C:min","C#:min","D:min","Eb:min","E:min","F:min","F#:min","G:min","Ab:min","A:min","Bb:min","B:min"]

def multivariate_Gaussian (x, MU, COV):
	return (exp(-0.5*np.dot((np.dot(np.transpose(x-MU),np.linalg.inv(COV))),(x-MU))))/sqrt(np.linalg.det(2*pi*COV))

def transition_Matrix ():
	T = np.empty([24,24])
	T[0][0] = f(12)
	T[1][0] = f(2)
	T[2][0] = f(8)
	T[3][0] = f(6)
	T[4][0] = f(4)
	T[5][0] = f(10)
	T[6][0] = f(0)
	T[7][0] = f(10)
	T[8][0] = f(4)
	T[9][0] = f(6)
	T[10][0] = f(8)
	T[11][0] = f(2)
	T[12][0] = f(5)
	T[13][0] = f(5)
	T[14][0] = f(9)
	T[15][0] = f(1)
	T[16][0] = f(11)
	T[17][0] = f(3)
	T[18][0] = f(7)
	T[19][0] = f(7)
	T[20][0] = f(3)
	T[21][0] = f(11)
	T[22][0] = f(1)
	T[23][0] = f(9)
	for i in range (0,24):
		T[:,i] = np.roll(T[:,0], i)
	return T

def mean_Matrix ():
	MU = np.zeros([12,24])
	MU[0][0] = MU[4][0] = MU[7][0] = 1
	for i in range (0,12):
		MU[:,i] = np.roll(MU[:,0], i)
	MU[0][12] = MU[3][12] = MU[7][12] = 1
	for i in range (12,24):
		MU[:,i] = np.roll(MU[:,12], i-12)
	return MU

def cov_Matrix ():
	COV = np.zeros([24,12,12])
	np.fill_diagonal(COV[0,:],0.2)
	COV[0][0][0] = COV[0][4][4] = COV[0][7][7] = 1
	COV[0][4][0] = COV[0][0][4] = 0.6
	COV[0][7][0] = COV[0][0][7] = 0.8
	COV[0][4][7] = COV[0][7][4] = 0.8
	for i in range (1,12):
		np.fill_diagonal(COV[i,:],0.2)
		COV[i,:,i] = np.roll(COV[0,:,0], i)
		COV[i,:,(i+4)%12] = np.roll(COV[0,:,4], i)
		COV[i,:,(i+7)%12] = np.roll(COV[0,:,7], i)
	np.fill_diagonal(COV[12,:],0.2)
	COV[12][0][0] = COV[12][3][3] = COV[12][7][7] = 1
	COV[12][3][0] = COV[12][0][3] = 0.6
	COV[12][7][0] = COV[12][0][7] = 0.8
	COV[12][3][7] = COV[12][7][3] = 0.8
	for i in range (13,24):
		np.fill_diagonal(COV[i,:],0.2)
		COV[i,:,i-12] = np.roll(COV[12,:,0], i-12)
		COV[i,:,(i-12+3)%12] = np.roll(COV[12,:,3], i-12)
		COV[i,:,(i-12+7)%12] = np.roll(COV[12,:,7], i-12)
	return COV

def init_Matrix ():
	return np.repeat(1.0/24,24)

def emission_Matrix ():
	T = Y.shape[1]
	B = np.zeros([24,T])
	for i in range (0,24):
		for j in range (0,T):
			B[i][j] = multivariate_Gaussian(Y[:,j],MU[:,i],COV[i])
	return B

def Viterbi ():
	K = 24;
	T = Y.shape[1]
	T1 = np.empty([K,T])
	T2 = np.empty([K,T])
	for i in range (0,24):
		T1[i][0] = log(P[i]) + log(B[i][0])
		T2[i][0] = 0
	for i in range (1,T):
		for j in range (0,K):
			m = 0
			for k in range (0,K):
				if (T1[k][i-1] + log(A[k][j]) > T1[m][i-1] + log(A[m][j])):
					m = k 
			T1[j][i] = log(B[j][i]) + T1[m][i-1] + log(A[m][j])
			T2[j][i] = m
	z = np.zeros(T)
	X = [None] * T
	z[T-1] = np.argmax(T1[:,T-1])
	X[T-1] = S[int(z[T-1])]
	for i in range (T-1,0,-1):
		z[i-1] = T2[int(z[i])][i]
		X[i-1] = S[int(z[i-1])]
	return np.asarray(X)

def getChromaBeats ():
	x, fs = librosa.load(sys.argv[1])
	audio = MP3(sys.argv[1])
	dur = audio.info.length
	window_size = 2210
	hop_size = 1105
	x_chroma = librosa.feature.chroma_stft(y=x, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=window_size)
	tempo,beats = librosa.beat.beat_track(y=x, sr=fs, onset_envelope=None, hop_length=hop_size, start_bpm=120.0, tightness=100, trim=False, bpm=None, units='frames')
	B = librosa.beat.beat_track(y=x, sr=fs, onset_envelope=None, hop_length=hop_size, start_bpm=120.0, tightness=100, trim=False, bpm=None, units='time')[1]
	b = beats.shape[0]
	T = x_chroma.shape[1]
	if (beats[0] > 0):
		x_beat_chroma = np.mean(x_chroma[:,0:beats[0]],axis=1)
	else:
		x_beat_chroma = np.zeros([12])
	for i in range (0,b-1):
		temp = np.mean(x_chroma[:,beats[i]:beats[i+1]],axis=1)
		if (beats[i+1] - beats[i] <= 0):
			continue
		x_beat_chroma = np.vstack((x_beat_chroma,temp))
	temp = np.mean(x_chroma[:,beats[i+1]:T],axis=1)
	x_beat_chroma = np.vstack((x_beat_chroma,temp))
	beats = np.concatenate((beats, np.array([T])))
	B = np.concatenate((B, np.array([dur])))
	return (np.transpose(x_beat_chroma),B)

Y,beats = getChromaBeats ()
S = stateSpace()
P = init_Matrix()
A = transition_Matrix()
MU = mean_Matrix()
COV = cov_Matrix()
B = emission_Matrix()
X = Viterbi()

for d, c in zip(beats, X):
    print d, c
