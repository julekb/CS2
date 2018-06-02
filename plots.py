import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import Counter
from librosa.display import waveplot
from librosa.feature import mfcc

import functions as f
import preprocessing as prep

plot_path = 'plots/'


def get_all_labels():

    pass


name = '_all'


if __name__ == '__main__':

    print('Here come some cool plots')

    ys = f.load_ys(name=name)
    Xs = f.load_Xs(name=name)
    rate = prep.rate

    # occurances in categories

    plt.figure(figsize=(12, 7))
    ys_occurances = Counter(ys)
    plt.bar(ys_occurances.keys(), ys_occurances.values())
    plt.xlabel('Categories')
    plt.ylabel('Occurances')
    plt.xticks(np.arange(len(ys_occurances.keys())), ys_occurances.keys(), fontsize=10, rotation=30)
    plt.savefig(plot_path + 'categories_in_dataset.jpg')

    # filtered / not filtered signal

    plt.clf()
    signal = Xs[0]
    signal_filtered = prep.butter_bandpass_filter(signal, lowcut=100, highcut=2000, fs=rate)
    plt.figure(figsize=(12, 5))
    waveplot(signal, sr=rate, alpha=0.7)
    waveplot(signal_filtered, sr=rate, alpha=0.7)
    plt.ylabel('Amplitude')
    plt.legend(['not filtered', 'filtered'])
    plt.savefig(plot_path + 'filtered_signal.jpg')

    # filtered / not filtered MFCC

    mfcc_signal = np.mean(mfcc(y=signal, sr=rate, n_mfcc=prep.NUM_mfcc).T, axis=0)
    mfcc_filtered = np.mean(mfcc(y=signal_filtered, sr=rate, n_mfcc=prep.NUM_mfcc).T, axis=0)

    print(mfcc_signal)
    plt.clf()
    plt.figure(figsize=(12, 5))
    # without first two components
    plt.plot(mfcc_signal.T[2:])
    plt.plot(mfcc_filtered.T[2:])
    plt.ylabel('Amplitude')
    plt.legend(['not filtered', 'filtered'])
    plt.savefig(plot_path + 'filtered_mfcc.jpg')