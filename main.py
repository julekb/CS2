import numpy as np
# import pandas as pd
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft, ifft
# from sklearn.model_selection import train_test_split


# Import data

# with open('pkl/Xs_mfcc.pkl', 'rb') as f:
#     Xs_mfcc = pkl.load(f)
# with open('pkl/Ys.pkl', 'rb') as f:
#     ys_num = pkl.load(f)


import re
import pickle as pkl
with open('pkl/scripts.pkl', 'rb') as f:
    script = pkl.load(f)

if __name__ == '__main__':
    c = 'evaluative'
    script = script[22]
    r = re.compile('.*'+c)

   
    print(script[script['AK'].str.contains(r, regex=True)])

    pass


