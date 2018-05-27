import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy.fftpack import fft, ifft
from sklearn.preprocessing import LabelBinarizer, scale
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
        r = re.compile('.*'+c)
        mask = script['AK'].str.contains(r, regex=True)
        mask = mask.fillna(False)
        subscript = script[mask]

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


def get_scripts(short=False):
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
    if short:
        script_names = script_names[:short]
        audio_name = audio_names[:short]

    for i, (script_name, audio_name) in enumerate(zip(script_names, audio_names)):

        script, audio = extract_files(script_path+script_name, path+audio_name)
        scripts.append(script)
        audios.append(audio)
        print (i+1, ' done')

    try:
        with open('pkl/scripts_part1.pkl', 'wb') as f:
            pkl.dump(scripts[:20], f)
        with open('pkl/scripts_part2.pkl', 'wb') as f:
            pkl.dump(scripts[20:], f)
        with open('pkl/audios_part1.pkl', 'wb') as f:
            pkl.dump(audios[:20], f)
        with open('pkl/audios_part2.pkl', 'wb') as f:
            pkl.dump(audios[20:], f)
    except:
        print('Not saved.')

    return scripts, audios


def get_parts(scripts2, audios2, name=''):
    print('Audio partition started.')

    Xs, ys = np.array([]), np.array([])
    for i, (script, audio) in enumerate(zip(scripts2, audios2)):

        X, Y = audio_partition(audio, rate, script, classes)
        print(X[0].shape, X.shape)

        ys = np.concatenate([ys, Y])
        try:
            Xs = np.concatenate([Xs, X])
        except:
            for x in X:
                Xs = np.append(Xs, x)
        print(i, ' done.')

    print('Saving Xs and ys.')

    try:  # file can be to big pkl.loads or cPickle could help
        with open('pkl/Xs'+name+'.pkl', 'wb') as f:
            pkl.dump(Xs, f)
        with open('pkl/Ys'+name+'.pkl', 'wb') as f:
            pkl.dump(ys, f)
    except:
        print('Not saved.')

    return Xs, ys


def get_mfccs(Xs, ys, NUM_mfcc, name=''):
    Xs_mfcc = np.empty((len(Xs), NUM_mfcc))
    empty_Xs = []

    for i, X in np.ndenumerate(Xs):
    #     X = np.fft.hfft(X) # Hermitian FFT gives a real output but the signal should have Hermitian symmetry?!
        
        try: 
            Xs_mfcc[i] = np.mean(librosa.feature.mfcc(y=X, sr=rate, n_mfcc=NUM_mfcc).T, axis=0)  # mean over time
            #normalization
            Xs_mfcc[i] = scale(Xs_mfcc[i], axis=1)
        except:
            empty_Xs.append(i)


    lb = LabelBinarizer().fit(ys)
    ys_num = lb.transform(ys)
    print(len(empty_Xs), len(Xs))
    Xs = np.delete(Xs, empty_Xs, 0)
    ys = np.delete(ys, empty_Xs, 0)
    print('Saving mfccs.')
    with open('pkl/Xs_mfcc'+name+'.pkl', 'wb') as f:
        pkl.dump(Xs_mfcc, f)
    with open('pkl/ys_num'+name+'.pkl', 'wb') as f:
        pkl.dump(ys_num, f)

    return Xs_mfcc, ys_num

def get_ffts(Xs, ys, NUM_ffts, name=''):
    Xs_ffts = np.empty((len(Xs), NUM_ffts))
    empty_Xs = []

    for i, X in np.ndenumerate(Xs):
    #     X = np.fft.hfft(X) # Hermitian FFT gives a real output but the signal should have Hermitian symmetry?!
        
        try: 
            Xs_ffts[i] = np.fft.fft(y=X, n_mfcc=NUM_ffts)  # mean over time
            #normalization
            # Xs_mfcc[i] = scale(Xs_mfcc[i], axis=1)
        except:
            empty_Xs.append(i)


    lb = LabelBinarizer().fit(ys)
    ys_num = lb.transform(ys)
    print(len(empty_Xs), len(Xs))
    Xs = np.delete(Xs, empty_Xs, 0)
    ys = np.delete(ys, empty_Xs, 0)
    print('Saving ftts.')
    with open('pkl/Xs_ffts'+name+'.pkl', 'wb') as f:
        pkl.dump(Xs_ffts, f)
    with open('pkl/ys_num'+name+'.pkl', 'wb') as f:
        pkl.dump(ys_num, f)

    return Xs_ffts, ys_num


def get_in_range(*args, start=0, end=-1):
    return tuple([l[start:end] for l in args])
"""
Data preprocessing and saving them to a pickle files so they can be opened in main.py.

"""

MFCC = True
FFT = True
N_PROC = 4
NUM_mfcc = 50
NUM_ffts = 75
rate = 22050
name = '_all'

classes = ['informative', 'evaluative', 'argumentative', 'directive', 'elicitative', 'affirmative', 'negative']


if __name__ == '__main__':


    # Import data

    print('get_scripts')
    scripts, audios = get_scripts()


    #  """
    
    with open('pkl/scripts_part1.pkl', 'rb') as f:
        scripts1 = pkl.load(f)
    with open('pkl/scripts_part2.pkl', 'rb') as f:
        scripts2 = pkl.load(f)
    with open('pkl/audios_part1.pkl', 'rb') as f:
        audios1 = pkl.load(f)
    with open('pkl/audios_part2.pkl', 'rb') as f:
        audios2 = pkl.load(f)

    scripts = np.hstack([scripts1, scripts2])
    audios = np.hstack([audios1, audios2])
    
    # """
    print('get_parts')
    scripts, audios = get_in_range(scripts, audios)
    Xs, ys = get_parts(scripts, audios, name=name)
    print(Xs.shape, ys.shape)

    #  """
    
    print('loading data')
    with open('pkl/Xs'+name+'.pkl', 'rb') as f:
        Xs = pkl.load(f)
    with open('pkl/Ys'+name+'.pkl', 'rb') as f:
        ys = pkl.load(f)

    print(Xs.shape, ys.shape)
    # Xs, ys = get_in_range(Xs, ys, start=0, end=40)
    # delete  empty rows

    # empty_inds = [i for i, x in np.ndenumerate(Xs) if x.size == 0]
    # Xs2 = np.delete(Xs, empty_inds)
    # ys2 = np.delete(ys, empty_inds)
    
    print('get_mfccs')
    Xs_mfcc, ys_num = get_mfccs(Xs, ys, NUM_mfcc, name)
    print('get_ftts')
    Xs_ftt, y_num = get_ffts(Xs, ys, NUM_ffts, name)

