import pickle as pkl
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras import regularizers
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from functions import *

"""
Fully connected neural network in keras.
Based on: https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/
"""


def neural_network(Xs_mfcc, ys_num, batch_size=32, epochs=100,
    learning_rate=0.01, patience=10):

    X_train, X_test, y_train, y_test = train_test_split(Xs_mfcc, ys_num, test_size=0.2, random_state=43)
    # TODO this should be done before
    # X_train = X_train[:, 7:35]
    # X_test = X_test[:, 7:35]

    num_labels = y_train.shape[1]

    # build model
    model = Sequential()

    model.add(Dense(50, input_shape=(X_train.shape[1],))) #256 ## , kernel_regularizer=regularizers.l2(0.1)
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    callback = EarlyStopping(monitor='val_acc', patience=patience)
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model_arguments = {
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'optimizer': optimizer,
    }

    model.compile(**model_arguments)

    cw = class_weight.compute_class_weight('balanced', np.unique(binary_to_categorical(y_train)), binary_to_categorical(y_train))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
        shuffle=True, class_weight=cw, callbacks=[callback])

    # TODO save trained model
    y_pred = model.predict(X_test)

    # confusion matrix
    print(confusion_matrix(binary_to_categorical(y_test), binary_to_categorical(y_pred)))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plot11.jpg')
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plot22.jpg')


def feature_selection(Xs_mfcc, ys_num, threshold):
    sel = VarianceThreshold(threshold=threshold)

    Xs_mfcc = sel.fit_transform(Xs_mfcc)
    print(Xs_mfcc.shape)

    return sel.fit_transform(Xs_mfcc)


def naive_categorization(ys_num):
    return Counter(ys_num)


def encode_ys(ys_num):
    ys_num = LabelEncoder().fit_transform(ys_num)
    ys_num = LabelBinarizer().fit_transform(ys_num)
    return ys_num


if __name__ == '__main__':

    seed(42)
    set_random_seed(41)
    name = '-negative-directive-affirmative'

    # loading files created with preprocessing.py

    with open('pkl/Xs_mfcc' + name + '.pkl', 'rb') as f:
        Xs_mfcc = pkl.load(f) #  should the first feature be deleted? it was somewhere written that the first one is just a coef
    with open('pkl/ys_num' + name + '.pkl', 'rb') as f:
        ys_num = pkl.load(f)

    # Xs_mfcc = feature_selection(Xs_mfcc, ys_num, 10)

    # print(np.var(Xs_mfcc, axis=0))
    Xs_mfcc = normalize(Xs_mfcc, axis=0)  # TODO does it make any difference?!

    # print(np.var(Xs_mfcc, axis=0))
    setup = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.002,
        'patience': 50,
    }
    ys_num = encode_ys(ys_num)

    print(Xs_mfcc.shape, ys_num.shape)
    neural_network(Xs_mfcc, ys_num, **setup)
