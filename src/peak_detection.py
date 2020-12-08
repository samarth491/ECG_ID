"""
This script is used to detect the QRS complex for the given signal.
We have used the algorithm mentioned in the paper:
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm. 
    In: IEEE Transactions on Biomedical Engineering 
    BME-32.3 (1985), pp. 230–236.
"""

from scipy.signal import butter, lfilter 
import numpy as np


"""
This function simulates the moving window integration according to
the window size provided.
params:
    arr -> input array we want to apply integration on
    window -> the window size for the integration
return:
    array after integration
"""

def cumulate(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    
    for i in range(1,window):
        ret[i-1] = ret[i-1] / i

    ret[window - 1:]  = ret[window - 1:] / window
    return ret


"""
This function is the actual algorithm that adaptively detects the
QRS complex.
params:
    detection -> it contains the array received from the `cumulate` 
                 function
    fs -> sample frequency of the signal
return:
    indices of peaks
"""

def pan_peak_detect(detection, fs):
    min_distance = int(0.25*fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i>0 and i<len(detection)-1:
            if detection[i-1]<detection[i] \
               and detection[i+1]<detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak]>threshold_I1 \
                   and (peak-signal_peaks[-1])>0.3*fs:
                        
                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125*detection[signal_peaks[-1]] \
                           + 0.875*SPKI
                    if RR_missed!=0:
                        if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                            missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak-signal_peaks[-2]>min_distance \
                                   and signal_peaks[-1]-missed_peak>min_distance \
                                   and detection[missed_peak]>threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2)>0:           
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak   

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1

                if len(signal_peaks)>8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66*RR_ave)

                index = index+1      
    
    signal_peaks.pop(0)

    return signal_peaks


"""
This is the driver function to find the QRS complex. Following is 
done to achieve the desired output:
-> Pass the signal through a BPF with cutoff freq as 5,15 Hz.
-> Differentiate the above signal.
-> Square the signal to amplify it.
-> Apply moving window integration on the signal.
-> Apply main algorithm to obtain indices of peaks

params:
    signal -> denoised ecg signals whose peaks are to be detected
    fs -> sampling freq of the signal
return:
    indices of peaks
"""

def detect_peaks(signal, fs):
    f_low = 5/fs
    f_high = 15/fs

    b, a = butter(1, [f_low*2, f_high*2], btype='bandpass')
    filtered_ecg = lfilter(b, a, signal)
    
    diff = np.diff(filtered_ecg) 
    squared = diff*diff

    N = int(0.12*fs)
    mwa = cumulate(squared, N)
    mwa[:int(0.2*fs)] = 0

    mwa_peaks = pan_peak_detect(mwa, fs)

    return mwa_peaks