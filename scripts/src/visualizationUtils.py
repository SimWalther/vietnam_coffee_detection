import matplotlib.pyplot as pl
from seaborn import FacetGrid, boxplot
import rasterio
import numpy as np


def draw_bands_boxplots(df, hue, title):
    """
    draw boxplots of data in a given dataframe
    :param df: the dataframe
    :param hue: colors scheme
    :param title: the boxplot title
    """

    g = FacetGrid(df, col="bands", col_wrap=4)
    g.fig.suptitle(title)
    g.map_dataframe(boxplot, x="bands", y="value", hue=hue)
    g.add_legend()
    g.set_titles(col_template="{col_name}")
    g.set(xticklabels=[])


def display_cross_val_map_class(fold_sets, map_shape, title, legends=["Other", "Test"], xlim=[106,110], ylim=[10, 16], figsize=(12, 6)):
    """
    code adapted from Romain Capocasale Master thesis display_cross_val_map_class
    :param fold_sets:
    :param map_shape:
    :param title:
    :param legends:
    :param xlim:
    :param ylim:
    :param figsize:
    :return:
    """

    fig, ax = pl.subplots(figsize=figsize)
    map_shape.plot(ax=ax, facecolor='Grey', edgecolor='k', alpha=0.5, linewidth=0.3)

    ax.set_prop_cycle(pl.cycler('color', ['tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:purple', 'tab:brown']))

    for fold_set in fold_sets:
        fold_set.plot(ax=ax, markersize=1, categorical=True, legend=True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    legend = ax.legend(legends)

    for handle in legend.legendHandles:
        handle.set_sizes([30])

    fig.suptitle(title)


def plot_confusion_matrix(confmatrix, labels_names, ax=None):
    """
    This function generates a colored confusion matrix.
    Code has been taken and adapted from plot_confusion_matrix of MLG course
    :param confmatrix: the confusion matrix
    :param labels_names: labels names
    :param ax: matplotlib axes object to plot to
    """
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


def label_first_detections(raster_path, dataframe, label_index, label_coordinates, image_size=9):
    """
    Find a specific label first detection at a given coor
    :param raster_path:
    :param dataframe:
    :param label_index:
    :param label_coordinates:
    :param image_size:
    :return:
    """
    with rasterio.open(raster_path) as raster:
        row_col_coffee_labels = [
            find_image_row_col_label_coordinate(raster, coord[0], coord[1], image_size) for coord in label_coordinates
        ]

    label_df = dataframe[dataframe['label_predicted'] == label_index]

    first_detections = []

    for row, col in row_col_coffee_labels:
        detections = label_df.loc[(label_df['image row'] == row) & (label_df['image col'] == col)]
        first_detections.append((row, col, detections.year.agg("min")))

    return first_detections


def find_image_row_col_label_coordinate(raster, lat, lon, image_size):
    """
    Find image row col from its coordinates
    :param raster: the raster from which the image has been taken
    :param lat: latitude
    :param lon: longitude
    :param image_size: the image size. Currently acts as step size too
    :return: the image row col
    """
    row, col = raster.index(lat, lon, precision=23)
    return round(row / image_size), round(col / image_size)
