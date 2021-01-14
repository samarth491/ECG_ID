"""
This script is used to train our CNN model on the database. The script output
is verbose and tells everything about the model like its architecture, live
accuracy and other parameters of each epoch and so on.
"""

import tensorflow as tf
import keras 
from keras.layers import Dense, Convolution1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from livelossplot import PlotLossesKeras


"""
This function creates the architecture of the CNN model.
params:
	feature -> no. of neurons in input layer
	depth -> hwow deep the network is
	out -> no. of neurons in output layer
return:
	the compiled model
"""

def make_model(feature, depth, out):
    model = Sequential()
    model.add(Convolution1D(32, 5, activation = 'relu', input_shape = (feature, depth)))     
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(128, 3, activation = 'relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(out, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auc')])
    return model


"""
This function is used to train our CNN model on the input data and
then save the model for further use.
"""

def train_cnn():
	df = pd.read_csv('../normalised_data.csv')
	N = len(df.columns) - 1
	N = str(N)

	df[N]=df[N].apply(str)

	print("============================= DataFrame Loaded ============================")

	X = df.iloc[:, :-1]
	y = df.iloc[:, -1:]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

	X_train = np.expand_dims(X_train, 2)
	X_test = np.expand_dims(X_test, 2)

	y_train = pd.get_dummies(y_train)
	y_test = pd.get_dummies(y_test)

	_, feature, depth = X_train.shape
	number_of_patients = y_train.shape[1]
	model = make_model(feature, depth, number_of_patients)

	print("\n======================== Model Architecture Defined =======================\n")

	model.summary()
	num_epochs=10

	history = model.fit(X_train, y_train, epochs = num_epochs, callbacks = [PlotLossesKeras()], validation_split = 0.25)
	score = model.evaluate(X_test, y_test, verbose = 1)

	model.save('../trained_model') 

	print("\n============================ Training Completed ===========================\n")


if __name__ == "__main__":
	train_cnn()