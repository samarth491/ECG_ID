import copy
import math
import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import CONSTANTS as const


def calc_energy(layer):
    res = 0.0
    for coff in layer:
        res += coff*coff
    
    res = np.log(res)
    return res


def calc_threshold(layer, n):
    arr = np.absolute(layer)
    md = np.median(arr)

    sigma = md/0.6745
    res = sigma * np.sqrt(2.0 * np.log10(n))

    return res


def smooth(signal):
    res = []
    n = len(signal)
    lmaxs = []
    upmins = []
    
    for i in range(1,n-1):
        pre = signal[i-1]
        cur = signal[i]
        nex = signal[i+1]

        if cur < pre and cur < nex:
            upmins.append(cur)
        if cur > pre and cur > nex:
            lmaxs.append(cur)

    lmax = np.amin(lmaxs)
    upmin = np.amax(upmins)

    if lmax < upmin:
        lmax, upmin = upmin, lmax

    res.append(signal[0])
    for i in range(1,n-1):
        pre = signal[i-1]
        cur = signal[i]
        nex = signal[i+1]

        if upmin <= cur and cur <= lmax:
            if (upmin <= pre and pre <= lmax) and (upmin <= nex and nex <= lmax):
                res.append((pre+cur+nex)/3)
            if (upmin <= pre and pre <= lmax) and (not (upmin <= nex and nex <= lmax)):
                res.append((pre+cur)/2)
            if (not (upmin <= pre and pre <= lmax)) and (upmin <= nex and nex <= lmax):
                res.append((nex+cur)/2)
            if (not (upmin <= pre and pre <= lmax)) and (not (upmin <= nex and nex <= lmax)):
                res.append(cur)
        else:
            res.append(cur)

    res.append(signal[n-1])
    return res


def snr(ori, fin):
    a=0; b=0

    for i in range(0,len(ori)):
        a += ori[i]*ori[i]
        b += (fin[i]-ori[i])*(fin[i]-ori[i])
    
    res = 10 * np.log(a/b)
    return res


filename = "114.csv"
dataset = np.genfromtxt(const.CSV_PATH+filename, delimiter=",", names=["time","lead1","lead2"], skip_header=1)
time = dataset["time"]
signal = dataset["lead2"]


fs = 360
seconds_to_plot = 5
samples_to_plot = fs * seconds_to_plot


s_avg = np.mean(signal, dtype=np.float64)
signal = signal - s_avg

plt.figure(1)
plt.subplot(211)
plt.plot(signal[:samples_to_plot])

layers = 10
wavelet = "db5"
coffs = pywt.wavedec(signal, wavelet, level=layers)

energy = []
thresh = []

for layer in coffs[1:]:
    energy.append(calc_energy(layer))
    thresh.append(calc_threshold(layer, len(signal)))

energy.reverse()
mx_loc = -1
mn_loc = -1

for i in range(1, layers-1):
    if (energy[i] > energy[i-1]) and (energy[i] > energy[i+1]) and mx_loc == -1:
        mx_loc = i+1

    if (energy[i] < energy[i-1]) and (energy[i] < energy[i+1]) and mn_loc == -1:
        mn_loc = i+1

layer = 0
if mx_loc != -1 and mn_loc != -1:
    if mx_loc < mn_loc:
        if mx_loc > 3:
            layer = (mx_loc + 1)/2
        else:
            layer = mx_loc
    else:
        if mn_loc > 3:
            layer = (mn_loc + 1)/2
        else:
            layer = mn_loc

if mx_loc == -1 and mn_loc != -1:
    if mn_loc > 3:
        layer = (mn_loc + 1)/2
    else:
        layer = mn_loc

if mx_loc != -1 and mn_loc == -1:
    if mx_loc > 3:
        layer = (mx_loc + 1)/2
    else:
        layer = mx_loc

if mx_loc == -1 and mn_loc == -1:
    layer = 3

layer = int(math.floor(layer))

plt.figure(2)
plt.plot(energy)


D_soft = copy.deepcopy(coffs)


for j in range(layers + 2 - layer, layers + 1):
    for pos in range(0, len(D_soft[j])):
        coff = D_soft[j][pos]

        if abs(coff) < thresh[j-1]:
            D_soft[j][pos] = 0
        else:
            D_soft[j][pos] = (np.sign(coff) * (abs(coff) - thresh[j-1]))


Y_soft = pywt.waverec(D_soft, wavelet)
Y_soft = smooth(Y_soft)

plt.figure(1)
plt.subplot(212)
plt.plot(Y_soft[:samples_to_plot])

snr_soft = (snr(signal, Y_soft))
print(snr_soft)

plt.show()