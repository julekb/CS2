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
from multiprocessing import Pool


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
        # should be with regrex and with checking AK2
        subscript = script[script['AK'] == c]

        for _, row in subscript.iterrows():
            start_time = row['begin']
            end_time = row['end']
            start = sum(x * float(t)
                        for x, t in zip([3600, 60, 1, 0.01], start_time.split(":"))) * rate
            end = sum(x * float(t)
                      for x, t in zip([3600, 60, 1, 0.01], end_time.split(":"))) * rate
            X.append(audio[int(start - 1):int(end)])
        Y = Y + [c] * len(subscript)

    return np.array(X), np.array(Y)

def extract_files(script_path, audio_path):

    script = pd.read_excel(script_path, index_col=0)
    # by default converted to mono and sr to 22050
    audio, _ = librosa.load(audio_path)
    return script, audio
    

"""
Data preprocessing and saving them to a pickle files so they can be opened in main.py.

"""

# flags
MFCC = True
FFT = True
N_PROC = 4



if __name__ == '__main__':

    rate = 22050
    classes = ['informative', 'evaluative', 'argumentative',
               'directive', 'affirmative', 'negative']
    # classes = ['informative', 'evaluative', 'argumentative', 'directive', 'elicitative', 'affirmative', 'negative']

    ### Import data ###

    #"""
    print('Data importing started.')

    path = 'wino_nagrania_final/'
    script_path = 'wino_nagrania_final/aktywności komunikacyjne/'

    file_names = [f for f in listdir(
        script_path) if isfile(join(script_path, f))]
    file_names2 = [f for f in listdir(path) if isfile(join(path, f))]
    r = re.compile("s\d{2}\-\d\.xlsx")
    script_names = list(filter(r.match, file_names))

    audio_names = []
    not_there = []
    for sn in script_names:
        r = re.compile(sn[0:5] + '.*\.mp3')
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

    script_names = script_names[:10]
    audio_name = audio_names[:10]


    for i, (script_name, audio_name) in enumerate(zip(script_names, audio_names)):
        # if i > 1:
        #     break
        script, audio = extract_files(script_path+script_name, path+audio_name)
        scripts.append(script)
        audios.append(audio)
        print (i+1, ' done')

    try:  # file can be to big pkl.loads or cPickle could help
        with open('pkl/scripts.pkl', 'wb') as f:
            pkl.dump(scripts, f)
        with open('pkl/audios.pkl', 'wb') as f:
            pkl.dump(audios, f)
    except:
        print('Not saved.')
    #"""
    with open('pkl/scripts.pkl', 'rb') as f:
        scripts = pkl.load(f)
    with open('pkl/audios.pkl', 'rb') as f:
        audios = pkl.load(f)
    

    # TODO
    # has to be done since there there is only one sample of the classes and it does not work
    # it should work with regex later
    
    try:
        scripts2 = np.delete(scripts, [10, 16])
        audios2 = np.delete(audios, [10, 16])
    except:
        scripts2, audios2 = scripts, audios

    print('Audio partition started.')


    Xs, ys = np.array([]), np.array([])
    for i, (script, audio) in enumerate(zip(scripts2, audios2)):
        X, Y = audio_partition(audio, rate, script, classes)
        Xs = np.concatenate([Xs, X])
        ys = np.concatenate([ys, Y])
        print(i, ' done.')


    """
    pool = Pool(processes=N_PROC)
    args = np.array([[audio, rate, script, classes] for (script, audio) in zip(scripts2, audios2)])
    print(args[0])
    results = pool.map_async(audio_partition, args)

    pool.close()
    pool.join()
    print(results.get())
    """

    print('Saving Xs and ys.')
    try:  # file can be to big pkl.loads or cPickle could help
        with open('pkl/Xs.pkl', 'wb') as f:
            pkl.dump(Xs, f)
        with open('pkl/Ys.pkl', 'wb') as f:
            pkl.dump(ys, f)
    except:
        print('Not saved.')
    

    print('loading data')
    with open('pkl/Xs.pkl', 'rb') as f:
        Xs = pkl.load(f)
    with open('pkl/Ys.pkl', 'rb') as f:
        ys = pkl.load(f)

    # delete  empty rows

    empty_inds = [i for i, x in np.ndenumerate(Xs) if x.size == 0]
    Xs2 = np.delete(Xs, empty_inds)
    ys2 = np.delete(ys, empty_inds)

    if MFCC:
        NUM_mfcc = 100
        Xs_mfcc = np.empty((len(Xs2), NUM_mfcc))

        # for i, X in np.ndenumerate(Xs2):
        #     X = np.fft.hfft(X) # Hermitian FFT gives a real output but the signal should have Hermitian symmetry?!
        #     Xs_mfcc[i] = np.mean(librosa.feature.mfcc(y=X, sr=rate, n_mfcc=NUM_mfcc).T, axis=0)

        lb = LabelBinarizer().fit(ys2)
        ys_num = lb.transform(ys2)
 

        print('Saving mfccs.')
        # with open('pkl/Xs_mfccF.pkl', 'wb') as f:
        #     pkl.dump(Xs_mfcc, f)
        # with open('pkl/ys_num.pkl', 'wb') as f:
        #     pkl.dump(ys_num, f)

    if FFT:
        pass

    # """

