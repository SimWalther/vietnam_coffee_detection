import numpy as np
from rasterio.plot import reshape_as_image
from math import inf
import matplotlib.pyplot as pl


# Adapted from split_dataset of MLG course
def split_dataset(dataset, train_test_ratio=0.8):
    np.random.shuffle(dataset)
    nb_train = int(len(dataset) * train_test_ratio)
    return dataset[:nb_train], dataset[nb_train:]


def images_from_dataset(dataset, bands):
    # min_per_band, max_per_band = compute_min_max_per_channel(dataset, bands)

    # Filter bands
    dataset = [
        [
            # normalize(img[1][band], min_per_band[i], max_per_band[i]) for i, band in enumerate(bands)
            img[1][band] for i, band in enumerate(bands)
        ] for img in dataset
    ]

    return np.array([reshape_as_image(img) for img in dataset])


def labels_from_dataset(dataset, labels):
    return np.array([labels.index(img[0]) for img in dataset])


# def compute_min_max_per_channel(dataset, bands):
#     min_per_channel = []
#     max_per_channel = []
#
#     for band in bands:
#         band_min = inf
#         band_max = -inf
#
#         for img in dataset:
#             img_values = img[1]
#
#             tmp_min = np.min(img_values[band])
#             tmp_max = np.max(img_values[band])
#
#             if tmp_min < band_min:
#                 band_min = tmp_min
#
#             if tmp_max > band_max:
#                 band_max = tmp_max
#
#         min_per_channel.append(band_min)
#         max_per_channel.append(band_max)
#
#     return min_per_channel, max_per_channel

#
# def normalize(values, min_val, max_val):
#     return (values - min_val) / (max_val - min_val)


# This function generates a colored confusion matrix.
# Adapted from plot_confusion_matrix of MLG course
def plot_confusion_matrix(confmatrix, labels_names, ax=None):
    if ax is None:
        ax = pl.subplot(111)

    ax.matshow(confmatrix, interpolation='nearest', cmap=pl.cm.Blues)

    for i in range(confmatrix.shape[0]):
        for j in range(confmatrix.shape[1]):
            ax.annotate(str(confmatrix[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8)
    ax.set_xticks(np.arange(confmatrix.shape[0]))
    ax.set_xticklabels([labels_names[label] for label in range(confmatrix.shape[0])], rotation='vertical')
    ax.set_yticks(np.arange(confmatrix.shape[1]))
    _ = ax.set_yticklabels([labels_names[label] for label in range(confmatrix.shape[1])])
    ax.set_xlabel('predicted label')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('true label')
