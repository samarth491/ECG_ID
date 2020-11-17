"""
This script is used to de-noise the ecg signals. It uses Discrete
Wavelet transform to perform the task. After applying DWT of 10 
level and mother wavelet `db5`, we find a level below which we need
to de-noise the signal. Then the signal is de-noised using soft
thresholding. Finally the signal is smotheened further to remove noise
left after previous steps.
"""

import copy

import numpy as np
import pywt

import load_data


"""
This function is used to calculate energy of each level of DWT.
params: 
    level -> the level in DWT whose cofficients are being taken
             into account
return:
    energy corresponding to the layer
"""

def calc_energy(level):
    res = 0.0
    for coff in level:
        res += coff*coff
    
    res = np.log(res)
    return res


"""
This function is used to calculate the threshold for each level.
params:
    level -> the level in DWT whose cofficients are being taken
             into account
    n -> number of samples
return:
    threshold value for each level
"""

def calc_threshold(level, n):
    tmp = np.absolute(level)
    md = np.median(tmp)

    sigma = md/0.6745
    res = sigma * np.sqrt(2.0 * np.log10(n))

    return res


"""
This function is used to find the level below which we need to denoise
the signal. First we find the first local max and first local min of 
the energies of each level. Then the max, min are used to find the
desired level.
params:
    energy -> list which contains energies of each level
return:
    the desired level below which we need to denoise the signal
"""

def find_denoise_level(energy):
    mx_loc = -1; mn_loc = -1
    n = len(energy)

    for i in range(1, n-1):
        if (energy[i] > energy[i-1]) and \
           (energy[i] > energy[i+1]) and mx_loc == -1:
            mx_loc = i+1

        if (energy[i] < energy[i-1]) and \
           (energy[i] < energy[i+1]) and mn_loc == -1:
            mn_loc = i+1

    res = 0

    # case 1: when both max and min exist
    if mx_loc != -1 and mn_loc != -1:
        if mx_loc < mn_loc: # when location of max is less than min 
            if mx_loc > 3:
                res = (mx_loc + 1)/2
            else:
                res = mx_loc
        else: # when location of min is less than max
            if mn_loc > 3:
                res = (mn_loc + 1)/2
            else:
                res = mn_loc

    #case 2: when only min exist
    if mx_loc == -1 and mn_loc != -1:
        if mn_loc > 3:
            res = (mn_loc + 1)/2
        else:
            res = mn_loc

    #case 3: when only max exist
    if mx_loc != -1 and mn_loc == -1:
        if mx_loc > 3:
            res = (mx_loc + 1)/2
        else:
            res = mx_loc

    #case 4: when both min and max do not exist
    if mx_loc == -1 and mn_loc == -1:
        res = 3

    res = int(np.floor(res))
    return res


"""
This function is used to smooth the signal to further remove noises.
params:
    signal -> signal which needs to be smotheened
return:
    smooth signal
"""

def smooth(signal):
    n = len(signal)
    res = []; lmaxs = []; upmins = []
    
    for i in range(1,n-1):
        pre = signal[i-1]
        cur = signal[i]
        nex = signal[i+1]

        if cur < pre and cur < nex:
            upmins.append(cur)
        if cur > pre and cur > nex:
            lmaxs.append(cur)

    # first we find the lowest local maxima
    # and highest local minima

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
            # case 1: when all 3 samples lie between lmax and upmin
            if (upmin <= pre and pre <= lmax) and \
               (upmin <= nex and nex <= lmax):
                res.append((pre+cur+nex)/3)
            
            # case 2: when nex does not lie between lmax and upmin
            if (upmin <= pre and pre <= lmax) and \
               (not (upmin <= nex and nex <= lmax)):
                res.append((pre+cur)/2)
            
            # case 3: when pre does not lie between lmax and upmin
            if (not (upmin <= pre and pre <= lmax)) and \
               (upmin <= nex and nex <= lmax):
                res.append((nex+cur)/2)
            
            # case 4: when none of the pre or nex lie between lmax and upmin
            if (not (upmin <= pre and pre <= lmax)) and \
               (not (upmin <= nex and nex <= lmax)):
                res.append(cur)

        else:
            res.append(cur)

    res.append(signal[n-1])
    return res


"""
This function is used to apply soft threshold to the cofficients of DWT
which are below the level found in `find_denoise_level` function. Afterwards,
it converts the coffs back to signal and returns the same
params:
    coffs -> cofficients of DWT
    levels -> levels of DWT
    denoise_loc -> level below which we need to denoise signal
    thresh -> list of thresholds of each level
    wavelet -> wavelet used as mother wavelet (This is used to reconstruct 
               signal)
return:
    denoised signal after applying soft threshold
"""

def soft_thresh(coffs, levels, denoise_loc, thresh, wavelet):
    soft_coff = copy.deepcopy(coffs)

    st = levels - denoise_loc + 2
    en = levels + 1

    for i in range(st, en):
        for pos in range(0, len(soft_coff[i])):

            coff = soft_coff[i][pos]

            if abs(coff) < thresh[i-1]:
                soft_coff[i][pos] = 0
            else:
                soft_coff[i][pos] = (np.sign(coff) * (abs(coff) - thresh[i-1]))

    signal = pywt.waverec(soft_coff, wavelet)
    return signal


"""
This function is used to denoise the signals so that they can be processed 
easily by the model.
params: None
return: list of denoised signals
"""

def denoise(datasets):
    denoised = []

    for dataset in datasets:
        signal = dataset["lead1"]

        s_avg = np.mean(signal, dtype=np.float64)
        signal = signal - s_avg

        levels = 10
        wavelet = "db5"
        coffs = pywt.wavedec(signal, wavelet, level=levels)

        energy = []
        thresh = []

        for level in coffs[1:]:
            energy.append(calc_energy(level))
            thresh.append(calc_threshold(level, len(signal)))

        energy.reverse()

        denoise_loc = find_denoise_level(energy)

        signal = soft_thresh(coffs, levels, denoise_loc, thresh, wavelet)
        signal = smooth(signal)
        denoised.append(signal)

    return denoised