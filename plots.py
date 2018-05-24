import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

plot_path = 'plots/'


def plot_communicative_activities(ys, title):

    plt.clf()
    plt.hist(ys, bins=len(np.unique(ys)))
    plt.title(title)
    plt.ylabel('occurances')
    plt.xlabel('comunicative activities')
    plt.savefig(plot_path + title + '.jpg')


if __name__ == '__main__':

    print('Here comes some cool plots')

    with open('pkl/Ys.pkl', 'rb') as f:
        Ys = pkl.load(f)

    plot_communicative_activities(Ys, 'Occurances of CA')
