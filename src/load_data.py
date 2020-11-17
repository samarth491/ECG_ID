"""
This script is used to load the signal dataset from the data folder.
The data extracted is in csv format for easy conversion into matrix.
This script also gives the sampling frequency of data which is used
in pre-processing steps. 
"""


import os

import numpy as np
import wfdb as wf

import CONSTANTS as const

"""
This is the driver function that helps to load the data.
params: None
return: list of datasets
"""

def load_data():
    filenames = [file for file in os.listdir(const.CSV_PATH)]
    datasets = []
    freqs = []

    for filename in filenames:
        dataset = np.genfromtxt(const.CSV_PATH+filename, 
                                delimiter=",", names=["time","lead1","lead2"], 
                                skip_header=1)
        datasets.append(dataset)

        record = wf.rdrecord(const.RAW_PATH+filename[:-4])
        freqs.append(record.__dict__['fs'])

    
    return [freqs, datasets]