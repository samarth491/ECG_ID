"""
This script is used to preprocess the data before sending it to the
model. We remove the noise from signals, find peaks and then segment
the data into smaller frames. Finally we return those frames in form
of a dictionary.
"""

import load_data
import clean_signal
import peak_detection


"""
This function is used to segment the signal for a particular person
according to the peaks detected.
params:
    signal -> de-noised signal which we want to segment
    peaks -> indices of detected peaks of tha same signals
    slide -> decide if we want the segments as moving window or not
return:
    segmented data for a particular signal
"""

def segment(signal, peaks, slide=True):
    n = len(peaks)
    N = len(signal)
    one_seg = 2 # no. of heartbeats in one segment
    sample_window = 128 # no. of samples before and after the peaks

    frames = []

    if slide:
        for i in range(0, n - one_seg + 1):
            l = peaks[i]
            r = peaks[i + one_seg - 1]
            l = max(l - sample_window, 0)
            r = min(r + sample_window, N - 1)
            frames.append(signal[l: r + 1])
    else:
        for i in range(0, n, one_seg):
            if i + one_seg >= n:
                break

            l = peaks[i]
            r = peaks[i + one_seg - 1]
            frames.append(signal[l: r + 1])

    return frames


"""
This is the driver function which preprocess the raw data.
return:
    preprocessed data in form of a dictionary where key is the id of
    the signal and value is the segmented data.
"""

def preprocess():
    data = load_data.load_data()

    print("============================= Database loaded =============================")

    freqs = data[0]
    raw_signals = data[1]
    ids = data[2]

    signals = clean_signal.clean_signal(freqs, raw_signals)

    print("============================== Noises Removed =============================")

    preprocessed_data = {}

    for idx in range(0, len(signals)):
        signal = signals[idx]
        freq = freqs[idx]

        print("Detecting peaks in patient", idx + 1, "data", end = '\r')

        peaks = peak_detection.detect_peaks(signal, freq)

        segmented_data = segment(signal, peaks)

        preprocessed_data[ids[idx]] = segmented_data

    return preprocessed_data

