import pickle as pkl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split

"""
Fully connected neural network in keras.
Based on: https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
"""

def main(Xs_mfcc, ys_num):

    X_train, X_test, y_train, y_test = train_test_split(Xs_mfcc, ys_num, test_size=0.2, random_state=42)
    
    # TODO this should be done before
    X_train = X_train[:,7:40]
    X_test = X_test[:,7:40]


    num_labels = y_train.shape[1]
    filter_size = 2

    # build model
    model = Sequential()

    model.add(Dense(50, input_shape=(X_train.shape[1],))) #256 ## , kernel_regularizer=regularizers.l2(0.1)
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


    adam = Adam(lr=0.1)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # Now let us train our model
    model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), shuffle=True)
    # model.fit(X_train, y_train, batch_size=20, epochs=3, validation_data=(X_train, y_train))

    # TODO save trained model
	

if __name__ == '__main__':

    # loading files created with preprocessing.py

    with open('pkl/Xs_mfcc.pkl', 'rb') as f:
        Xs_mfcc = pkl.load(f)
    with open('pkl/Ys.pkl', 'rb') as f:
        ys_num = pkl.load(f)

    main(Xs_mfcc, ys_num)
