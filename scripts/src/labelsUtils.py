import shapefile
import shapely.geometry
import numpy as np
from enum import Enum


class Label(Enum):
    CACAO = 1
    COFFEE = 2
    COMPLEX_OIL = 3
    NATIVEVEGE = 4
    OIL_PALM = 5
    RUBBER = 6
    UNKNOWN = 7
    SEASONAL = 8
    URBAN = 9
    WATER = 10
    OTHER_TREE = 11
    OTHER_NO_TREE = 12
    NATIVE_NO_TREE = 13
    WATER_OTHER = 14
    PEPPER = 15
    CASSAVA = 16
    TEA = 17
    RICE = 18
    BANANA_JUNG = 19
    BABY_PALM = 20
    CUT_OFF_REGROW = 21
    NATURAL_WETLAND = 22
    INTERCROP = 23
    DECIDUOUS_FOREST = 24
    STICK_PEPPER = 25
    FLOODED_PLANTATION = 26
    PINE_TREES = 27
    COCONUT = 28
    BAMBOO = 29
    SAVANA = 30
    MANGO = 31
    OTHER_FRUIT_TREE_CROP = 32
    WATER_MINE = 33


class LabelCategory(Enum):
    CULTURE = 0
    FOREST = 1
    NATIVE_NO_TREE = 2
    URBAN = 3
    WATER = 4


def labels_coordinates_from_files(shapefiles_paths, boundaries):
    """
    Create a dictionary with coordinates of each entries for every labels
    :param shapefiles_paths: the path to the labels shapes
    :param boundaries: the boundaries outside which entries are excluded
    :return: the labels coordinates dictionary
    """

    # Create a dictionary which will contain all
    # points classified in the opened shapefiles
    # with their coordinates lat-lon
    labels_coordinates = {}

    # Initialize the dictionary
    for i in range(1, 34):
        labels_coordinates[i] = []

    # Add each points coordinate
    # in its corresponding class
    for path in shapefiles_paths:
        for shape_record in shapefile.Reader(path).shapeRecords():
            if shape_record.record.Class <= len(Label):
                current_list = labels_coordinates.get(shape_record.record.Class)

                # Add label coordinate only if this label is inside the boundaries
                if shapely.geometry.Point(shape_record.shape.points[0]).within(boundaries):
                    current_list.append(shape_record.shape.points[0])
                    labels_coordinates[shape_record.record.Class] = current_list

    return labels_coordinates


def category_from_label(label):
    if (
            label == Label.COFFEE or
            label == Label.RUBBER or
            label == Label.PEPPER or
            label == Label.TEA or
            label == Label.RICE or
            label == Label.INTERCROP or
            label == Label.STICK_PEPPER or
            label == Label.SEASONAL
    ):
        return LabelCategory.CULTURE
    elif (
            label == Label.NATIVEVEGE or
            label == Label.OTHER_TREE or
            label == Label.DECIDUOUS_FOREST or
            label == Label.PINE_TREES
    ):
        return LabelCategory.FOREST
    elif label == Label.NATIVE_NO_TREE:
        return LabelCategory.NATIVE_NO_TREE
    elif label == Label.URBAN:
        return LabelCategory.URBAN
    elif label == Label.WATER:
        return LabelCategory.WATER
    else:
        return None


def categories_from_label_set(labels, label_set):
    return np.asarray([
        category_from_label(labels[label]).value for label in label_set
    ])
