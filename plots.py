import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import Counter

plot_path = 'plots/'


def plot_communicative_activities(ys, title):

    plt.clf()
    plt.hist(ys, bins=len(np.unique(ys)))
    plt.title(title)
    plt.ylabel('occurances')
    plt.xlabel('comunicative activities')
    plt.savefig(plot_path + title + '.jpg')

def get_all_labels():

    pass

name = '_all'

if __name__ == '__main__':

    print('Here comes some cool plots')

    with open('pkl/Ys'+name+'.pkl', 'rb') as f:
        Ys_all = pkl.load(f)

    plot_communicative_activities(Ys_all, 'All occurances of CA')
    print(Counter(Ys_all))
