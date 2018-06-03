import pickle as pkl
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from keras.callbacks import EarlyStopping
from collections import Counter
from keras.callbacks import ModelCheckpoint

from functions import *
from preprocessing import generate_data

"""
Fully connected neural network in keras.
Based on: https://www.analyticsvidhya.com/
blog/2017/08/audio-voice-processing-deep-learning/
"""


def neural_network(X_train, X_test, y_train, y_test, model_name='', batch_size=32, epochs=300,
    learning_rate=0.01, patience=30):

    seed(42)
    set_random_seed(41)


    # X_train = np.vstack([X_train, generate_data(X_train)])
    # y_train = np.vstack([y_train, y_train])
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

    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model_arguments = {
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'optimizer': optimizer,
    }

    model.compile(**model_arguments)

    # checkpoint
    
    filepath = 'models/NN-gridsearch-' + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    early_callback = EarlyStopping(monitor='val_acc', patience=100)
    callbacks_list = [checkpoint, early_callback]

    
    cw = class_weight.compute_class_weight('balanced', np.unique(binary_to_categorical(y_train)), binary_to_categorical(y_train))

    model_setup = {
        'batch_size': batch_size,
        'epochs': epochs,
        'validation_data': (X_test, y_test),
        'shuffle': True,
        'class_weight': cw,
        'callbacks': callbacks_list,
        'verbose': 0
    }
    print('batch_size', model_setup['batch_size'])

    history = model.fit(X_train, y_train, **model_setup)

    # TODO save trained model
    # model = load_model(filepath)
    # evalmodel = model.evaluate(X_test, y_test)
    
    # print(evalmodel)
    

    # confusion matrix
    # print(confusion_matrix(binary_to_categorical(y_test), binary_to_categorical(y_pred)))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/' + model_name + 'model_acc.jpg')
    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/model_loss.jpg')
    del model
    return filepath


def feature_selection(Xs_mfcc, ys_num, threshold):
    sel = VarianceThreshold(threshold=threshold)

    Xs_mfcc = sel.fit_transform(Xs_mfcc)
    print(Xs_mfcc.shape)

    return sel.fit_transform(Xs_mfcc)


def naive_categorization(ys_num):
    ys_dict = Counter(ys_num)
    max_value = max(ys_dict.values())
    max_key = ''
    for key, value in ys_dict.items():
        if value == max_value:
            max_key = key
            break
    values_sum = sum(ys_dict.values())
    print(max_value, values_sum)
    print('Most common: %s with %f probability of occurance' % (max_key, max_value / values_sum))


if __name__ == '__main__':


    name = '-negative-directive-affirmative'

    # loading files created with preprocessing.py

    Xs_mfcc = load_Xs_mfcc(name=name)
    ys_num = load_ys_num(name=name)
   
    Xs_mfcc = Xs_mfcc[:, 1:]
    Xs_mfcc = feature_selection(Xs_mfcc, ys_num, 10)

    Xs_mfcc = normalize(Xs_mfcc, axis=0)

    setup = {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.002,
        'patience': 150,
    }

    naive_categorization(ys_num)
    ys_num = encode_ys(ys_num)
    

    print(Xs_mfcc.shape, ys_num.shape)
    # neural_network(Xs_mfcc, ys_num, **setup)


