import numpy as np
import pandas as pd
import sys

def SW_len(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    H = np.empty((n1 + 1, n2 + 1))
    for i in range(0, n1 + 1):
        H[i][0] = 0
    for j in range(0, n2 + 1):
        H[0][j] = 0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if (s1[i-1][0] == s2[j-1][0]):
                simi = 4
            else:
                simi = -2
            H[i][j] = max(0, H[i-1][j-1]+simi, H[i-1][j]-1, H[i][j-1]-1)
    return np.amax(H)

df_true = pd.read_csv(sys.argv[1], header=None, sep=' ')
df_obs = pd.read_csv(sys.argv[2], header=None, sep=' ')

true_time_start = df_true[0].values.tolist()
true_time_end = df_true[1].values.tolist()
true_chord = df_true[2].values.tolist()
obs_time = df_obs[0].values.tolist()
obs_chord = df_obs[1].values.tolist()

T = len(true_chord)
O = len(obs_chord)

for i in range (0,T):
	tc = true_chord[i]
	if (tc == 'N'):
		continue
	z = tc.find(':')
	if (z != -1):
		if (not tc[z+1].isalpha()):
			tc = tc[0:z] + ":maj"
			true_chord[i] = tc
			continue
		if (tc[z+1] != 'm'):
			tc = tc[0:z] + ":maj"
		if (tc[z+2] == 'a'):
			tc = tc[0:z] + ":maj"
		if (tc[z+2] == 'i'):
			tc = tc[0:z] + ":min"
	z = tc.find('/')
	if (z != -1):
		tc = tc[0:z] + ":maj"
	if (len(tc) == 1):
		tc += ':maj'
	true_chord[i] = tc

obs_seq = [obs_chord[0]]
l = obs_chord[0]
for i in range (1,O):
	if (obs_chord[i] != l):
		obs_seq.append(obs_chord[i])
		l = obs_chord[i]

for i in range (0,T):
	if (true_time_end[i] > 60):
		break
T = i
visited = np.repeat(0,T)

true_seq = [true_chord[0]]
l = true_chord[0]
for i in range (1,T):
	if (true_chord[i] != l):
		true_seq.append(true_chord[i])
		l = true_chord[i]

c = 0
d = 0
for i in range (0,T):
	for j in range (0,O):
		if (obs_time[j] >= true_time_start[i] and obs_time[j] <= true_time_end[i]):
			if (true_chord[i] == 'N' or obs_chord[j] == true_chord[i]):
				c += 1
			else:
				d += 1

print c, c+d