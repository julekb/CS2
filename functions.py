import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


def binary_to_categorical(binary):

    """Function to convert binary labels into categorical."""

    cat = np.zeros(binary.shape[0])
    for i in range(binary.shape[0]):
        cat[i] = binary[i, :].argmax()
    return cat

def encode_ys(ys_num):
    ys_num = LabelEncoder().fit_transform(ys_num)
    ys_num = LabelBinarizer().fit_transform(ys_num)
    return ys_num


def load_ys_num(name='_all'):

    with open('pkl/Ys_num' + name + '.pkl', 'rb') as f:
        Ys_num = pkl.load(f)

    return Ys_num


def load_ys(name='_all'):

    with open('pkl/Ys' + name + '.pkl', 'rb') as f:
        Ys = pkl.load(f)

    return Ys

def load_ys_excluded():

    with open('pkl/ys_exlcuded-negative-directive-affirmative_final.pkl', 'rb') as f:
        Ys = pkl.load(f)

    return Ys


def load_Xs_num(name='_all'):

    with open('pkl/Xs_num' + name + '.pkl', 'rb') as f:
        Xs_num = pkl.load(f)

    return Xs_num


def load_Xs(name='_all'):

    with open('pkl/Xs' + name + '.pkl', 'rb') as f:
        Xs = pkl.load(f)

    return Xs

def load_Xs_filtered(name='_all'):

    with open('pkl/Xs_filtered' + name + '.pkl', 'rb') as f:
        Xs = pkl.load(f)

    return Xs


def load_Xs_mfcc(name='_all'):

    with open('pkl/Xs_mfcc' + name + '.pkl', 'rb') as f:
        Xs = pkl.load(f)

    return Xs


def load_scripts_audios(name=''):

    with open('pkl/scripts' + name + '.pkl', 'rb') as f:
        scripts = pkl.load(f)
    with open('pkl/audios' + name + '.pkl', 'rb') as f:
        audios = pkl.load(f)

    return scripts, audios


def load_scripts_audios2():
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
    return audios, scripts


def load_grid(name=''):

    with open('pkl/grid-search.pkl', 'rb') as f:
        return pkl.load(f)


def save(object, path):
    with open('pkl/' + path + '.pkl', 'wb') as f:
        pkl.dump(object, f)
    return


def load(path):

    with open('pkl/' + path + '.pkl', 'rb') as f:
        return pkl.load(f)

