"""
This script is used to convert raw physionet data to csv format. This will
help to ease the processing and fast loading of data.
First we import the directory and path to the raw data. Then the script reads
names of all records. These records are then read using wfdb and are converted
to csv format using `to_csv` function. For more information on how this
function works, please look into function description.
"""


import os

import wfdb as wf
import numpy as np
import pandas as pd

import CONSTANTS as const

"""
    Reads the records of a patient using wfdb - native Python 
    waveform-database package. The recordings are stored in key value 
    `p_signal` of dictionary `__dict__`. We use the frequency to find the 
    time-stamp realted to data for proper processing and plotting of data. 
    Finally this data is converted to csv format and stored under specified 
    file name.
    Note: The original data contains 25 hours long ECG recordings. However,
          here we have used 2 hours of data only.
    
    params: file name of the concerned record
    return: None
"""

def to_csv(file):
    dataset = const.RAW_PATH + file
    dataset_csv = const.CSV_PATH + file + ".csv"
    
    # make the folder to store csv files if it does not already exist
    if not os.path.exists(const.CSV_PATH):
        os.mkdir(const.CSV_PATH)
        
    record = wf.rdrecord(dataset)
    readings = record.__dict__['p_signal']
    
    # `readings` contains two colums containing ecg signals corresponding to
    # lead 1 and lead 2 respectively.

    freq = record.__dict__['fs']
    samples = 7200*freq
    
    time_stamp = np.zeros((samples,1))
    ecg_lead_1 = np.zeros((samples,1))
    ecg_lead_2 = np.zeros((samples,1))
    data = np.zeros((samples,3))
    
    for i in range(0, samples):
        time_stamp[i][0] = (i/freq)
        ecg_lead_1[i][0] = readings[i][0]
        ecg_lead_2[i][0] = readings[i][1]
        
        data[i][0] = time_stamp[i][0]
        data[i][1] = ecg_lead_1[i][0]
        data[i][2] = ecg_lead_2[i][0]
        
    pd.DataFrame(data).to_csv(dataset_csv, 
                              header=['timestamp', 'lead1', 'lead2'], 
                              index=False)


"""
Driver function used to convert raw data to csv format
"""

def raw_to_csv(): 
    # `files` contains names of all patients records
    files = [file[:-4] 
             for file in os.listdir(const.RAW_PATH) 
             if file.endswith('.dat')]
    
    for file in files:
        to_csv(file)
    

if __name__ == "__main__":
    raw_to_csv()