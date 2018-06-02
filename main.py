import preprocessing as prep


if __name__ == '__main__':

    NUM_mfcc = 50
    classes = ['informative', 'evaluative', 'argumentative', 'directive', 'elicitative', 'affirmative', 'negative']
    excluded_classes = ['negative', 'directive', 'affirmative']

    # preprocessing

    scripts, audios = prep.get_scripts()
    Xs, ys = prep.get_parts(scripts, audios, save=False)
    Xs = prep.get_filtered(Xs, save=False)
    Xs_mfcc = prep.get_mfccs(Xs, NUM_mfcc, name='_all', save=True)

    print('Data preprocessed.')

    # train, test set split

    # Neural network
