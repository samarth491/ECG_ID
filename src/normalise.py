"""
This script is used to normalise all the segents to a same size.
The segments initially are of different sizes, however, that can not
be fed to the CNN. Thus, we need to normalise the size of segments.
"""

import sys

import pandas as pd

import preprocess


"""
This function returns the size of the minimum segment among all the
patients.
params:
    data -> a dictionary conatining all the preprocessed data
return:
    size of minimum segment among all segments
"""

def find_size(data):
    min_size = sys.maxsize

    for patient in data:
        for segment in data[patient]:
            min_size = min(segment.size, min_size)

    return min_size        
            

"""
This function is used to trim all the segments to the size of minimum segment.
The segment is trimmed equally from both start and end to reduce biases.
params:
    segment -> segment we want to trim
    size -> size to which this segment has to be trimmed
return:
    trimmed segment
"""

def normalise(segment, size):
    if segment.size > size:
        start = ((segment.size) - size) // 2
        return segment[start : start + size]
    return segment


"""
This function is used to create a dataframe from the preprocessed data
which will be fed to the CNN.
params:
    data -> preprocessed data
    normalise_size -> size of the smallest segment
return:
    DataFrame that can be fed to the CNN model
"""

def create_df(data, normalised_size):
    df = pd.DataFrame(columns = [i for i in range(normalised_size + 1)])

    ptr = 0
    idx = 1
    for patient in data:
        print("Creating DataFrame for patient", idx, end = '\r')

        for ind in range(len(data[patient])):
            data[patient][ind] = normalise(data[patient][ind], normalised_size)
            df.loc[ptr] = data[patient][ind].tolist() + [patient]
            ptr += 1

            done = ((ind + 1) / len(data[patient])) * 100.0
            print("Creating DataFrame for patient", idx, "%6.2f %% done" %(done), end = '\r')
        
        idx += 1

    return df     


"""
This is the driver function that collects the preprocessed data,
finds its normalised size and then returns the created dataframe in
the form of a file.
"""

def get_normalised_data():
    data = preprocess.preprocess()

    print("========================= Preprocessing Completed =========================")

    normalised_size = find_size(data)

    df = create_df(data, normalised_size)
    df.to_csv('../normalised_data.csv', encoding = 'utf-8', index = False)

    print("========================= Normalization Completed =========================")

if __name__ == "__main__":
    get_normalised_data()