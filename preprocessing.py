import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pickle as pkl
from os import listdir
from os.path import isfile, join
import re




"""
Data preprocessing and saving them to a pickle files so they can be opened in main.py.

"""

def audio_partition(audio, rate, script, classes):
    """
    input:
    audio: audio file as an array
    rate: sampling rate of the audio file
    script: pandas dataframe
    classes: classes in which data should be divided
    
    output:
    X: array of parts of audio file
    Y: array of labels according to X
    """
    
    X, Y = [], []
    
    for c in classes:
        subscript = script[script['AK'] == c] #should be with regrex and with checking AK2

        for _, row in subscript.iterrows():
            start_time = row['begin']
            end_time = row['end']
            start = sum(x * float(t) for x, t in zip([3600, 60, 1, 0.01], start_time.split(":"))) * rate
            end = sum(x * float(t) for x, t in zip([3600, 60, 1, 0.01], end_time.split(":"))) * rate
            X.append(audio[int(start-1):int(end)])
        Y = Y + [c] * len(subscript)

    
    return np.array(X), np.array(Y)



if __name__ == '__main__':

    ### Import data ###
    print('Data importing started.')

    path = 'wino_nagrania_final/'
    script_path = 'wino_nagrania_final/aktywności komunikacyjne/'

    file_names = [f for f in listdir(script_path) if isfile(join(script_path, f))]
    file_names2 = [f for f in listdir(path) if isfile(join(path, f))]
    r = re.compile("s\d{2}\-\d\.xlsx")
    script_names = list(filter(r.match, file_names))

    rate = 22050

    audio_names = []
    not_there = []
    for sn in script_names:
        r = re.compile(sn[0:5]+'.*\.mp3')
        name = list(filter(r.match, file_names2))
        if name == []:
            not_there.append(sn)
        else:
            audio_names.append(name[0])
        
    for sn in not_there:
        script_names.remove(sn)

    print('Filenames found.')
        
    scripts = []
    audios = []

    for i, (script_name, audio_name) in enumerate(zip(script_names, audio_names)):
        script = pd.read_excel(script_path+script_name, index_col=0)
        audio, _ = librosa.load(path+audio_name) # by default converted to mono and sr to 22050
        scripts.append(script)
        audios.append(audio)
        print (i+1, ' done')


    # TODO
    # has to be done since there there is only one sample of the classes and it does not work
    # it should work with regex later
    scripts2 = np.delete(scripts, [10, 16])
    audios2 = np.delete(audios, [10, 16])


    print('Audio partition started.')
    Xs, ys = np.array([]), np.array([])
    for i, (script, audio) in enumerate(zip(scripts2, audios2)):
        X, Y = audio_partition(audio, rate, script, classes)
        Xs = np.concatenate([Xs, X])
        ys = np.concatenate([ys, Y])
        print(i, ' done.')


    print('Saving Xs and ys.')
    with open('pkl/Xs.pkl', 'wb') as f:
        pkl.dump(Xs, f)
    with open('pkl/Ys.pkl', 'wb') as f:
        pkl.dump(ys, f)

    # delete an empty row
    empty_inds = [i for i,x in np.ndenumerate(Xs) if x.size == 0]
    Xs2 = np.delete(Xs, empty_inds)
    ys2 = np.delete(ys, empty_inds)

    NUM_mfcc = 100
    Xs_mfcc = np.empty((len(Xs2), NUM_mfcc))

    for i, X in np.ndenumerate(Xs2):
        Xs_mfcc[i] = np.mean(librosa.feature.mfcc(y=X, sr=rate, n_mfcc=NUM_mfcc).T, axis=0)
        
    lb = LabelBinarizer().fit(ys2)
    ys_num = lb.transform(ys2)

    print('Saving mfccs.')
    with open('pkl/Xs_mfcc.pkl', 'wb') as f:
        pkl.dump(Xs_mfcc, f)
    with open('pkl/Ys.pkl', 'wb') as f:
        pkl.dump(ys_num, f)
    