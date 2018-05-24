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
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

from functions import *

"""
Fully connected neural network in keras.
Based on: https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
"""


def main(Xs_mfcc, ys_num):

    X_train, X_test, y_train, y_test = train_test_split(Xs_mfcc, ys_num, test_size=0.2, random_state=42)
    # TODO this should be done before
    X_train = X_train[:, 7:35]
    X_test = X_test[:, 7:35]

    num_labels = y_train.shape[1]

    # build model
    model = Sequential()

    model.add(Dense(100, input_shape=(X_train.shape[1],))) #256 ## , kernel_regularizer=regularizers.l2(0.1)
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(110))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


    adam = Adam(lr=0.1)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # Now let us train our model

    cw = class_weight.compute_class_weight('balanced', np.unique(binary_to_categorical(y_train)), binary_to_categorical(y_train))
    history = model.fit(X_train, y_train, batch_size=8, epochs=100, validation_data=(X_test, y_test),
        shuffle=True, class_weight=cw)
    # model.fit(X_train, y_train, batch_size=20, epochs=3, validation_data=(X_train, y_train))

    # TODO save trained model
    y_pred = model.predict(X_test)
    
    ### confusion matrix
    print(confusion_matrix(binary_to_categorical(y_test), binary_to_categorical(y_pred)))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plot3.jpg')
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plot32.jpg')

if __name__ == '__main__':

    # loading files created with preprocessing.py

    with open('pkl/Xs_mfcc.pkl', 'rb') as f:
        Xs_mfcc = pkl.load(f)
    with open('pkl/ys_num.pkl', 'rb') as f:
        ys_num = pkl.load(f)

    Xs_mfcc = normalize(Xs_mfcc, axis=0)  # TODO does it make any difference?!

    main(Xs_mfcc, ys_num)
