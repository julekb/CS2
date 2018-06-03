import preprocessing as prep
from functions import *
from NN import neural_network

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pickle as pkl
import matplotlib.pyplot as plt

if __name__ == '__main__':

    NUM_mfcc = 50
    classes = ['informative', 'evaluative', 'argumentative', 'directive', 'elicitative', 'affirmative', 'negative']
    excluded_classes = ['negative', 'directive', 'affirmative']

    # preprocessing

    """
    scripts, audios = prep.get_scripts()
    Xs, ys = prep.get_parts(scripts, audios, save=False)
    Xs = prep.get_filtered(Xs, save=False)
    Xs = prep.get_mfccs(Xs, NUM_mfcc, name='_all', save=True)

    # print('Data preprocessed.')
    
    Xs = load_Xs_mfcc(name='_all')
    ys = load_ys(name='_all')
    Xs, ys = prep.get_excluded(Xs, ys, excluded_classes, save=False)

    ys = encode_ys(ys)
    print(Xs)
    Xs = normalize(Xs)
    # train, test set split

    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.05, random_state=43)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=44)

    save([X_train, X_dev, X_test, y_train, y_dev, y_test], 'splited')
   """
    X_train, X_dev, X_test, y_train, y_dev, y_test = load('splited')

    # make up data

    # X_train = np.vstack([X_train, generate_data(X_train)])
    # y_train = np.vstack([y_train, y_train])

    # grid search

    batch_sizes = [8, 16, 32, 64, 128]
    learning_rates = [0.1, .05, 0.02, 0.01, 0.001]

    histories = []
    grid = []

    make_keras_picklable()

    i_max = len(batch_sizes) * len(learning_rates) + 1
    for i, batch_size in enumerate(batch_sizes):
        for j, learning_rate in enumerate(learning_rates):
            setup = {


                'batch_size': batch_size,
                'learning_rate': learning_rate, 
            }
            name = '-'.join([str(x) for x in list(setup.values())])
            setup2 = {
                'X_train': X_train,
                'X_test': X_dev,
                'y_train': y_train,
                'y_test': y_dev,
                'model_name': name,
                'epochs': 300,
            }
            setup.update(setup2)
            
            model_eval, history = neural_network(**setup)
            grid.append([i, j, model_eval])
            histories.append(history)
                
            print(i, ' done out of ', i_max)
            i += 1

    save(histories, 'histories')
    print(grid)
    save(grid, 'grid-search')
    

    # Neural network
