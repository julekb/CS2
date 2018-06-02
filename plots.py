import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from collections import Counter

plot_path = 'plots/'



def get_all_labels():

    pass

name = '_all'

if __name__ == '__main__':

    print('Here comes some cool plots')


    with open('pkl/Ys'+name+'.pkl', 'rb') as f:
        ys = pkl.load(f)

    # occurances in categories
    plt.figure(figsize=(12, 7))
    ys_occurances = Counter(ys)
    plt.bar(ys_occurances.keys(), ys_occurances.values())
    plt.xlabel('Categories')
    plt.ylabel('Occurances')
    plt.xticks(np.arange(len(ys_occurances.keys())), ys_occurances.keys(), fontsize=10, rotation=30)
    plt.savefig(plot_path + 'categories_in_dataset.jpg')

    