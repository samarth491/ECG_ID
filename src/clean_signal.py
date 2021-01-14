"""
This script is used to clean the noisy ECG recordings. This is a necessary
step as noisy data would affect the accuracy of the model. Thus both low
frequency baseline wander noise and high frequency noises like EMG, poweline
interference are removed.
"""

from scipy.signal import butter, filtfilt 

import denoise


"""
This function is used to remove baseline wander noise. It uses a HPF of order
4 to perform the task.
params:
    signal -> signal which needs to be cleaned
    fs -> sampling frequency of the signal
return:
    signal with baseline wander removed
"""

def remove_baseline_wander(signal, fs):
    nyq = 0.5 * fs
    cutoff = 0.5
    freq = cutoff / nyq
    order = 4

    b, a = butter(order, freq, 'high')
    res = filtfilt(b, a, signal)
    return res


"""
This function is used to clean ECG signals from all noises. High frequency
noise is removed using `denoise` function while low frequency noises are
removed using `remove_baseline_wander` function.
params:
    freqs -> list of sample frequencies of signals
    datasets -> dataset containing noisy signals
return:
    clean signals after applying filters
"""

def clean_signal(freqs, datasets):
    denoised_signals = denoise.denoise(datasets)

    print("======================= High Frequency Noise Removed ======================")


    clean_signals = []

    for idx in range(0, len(denoised_signals)):
        signal = denoised_signals[idx]
        fs = freqs[idx]

        print("Removing low frequency noise from patient", idx + 1, "data", end = '\r')

        clean_signals.append(remove_baseline_wander(signal, fs))

    return clean_signals
    
