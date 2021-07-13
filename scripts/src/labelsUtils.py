import shapefile
import shapely.geometry
import numpy as np
from enum import Enum


class Label(Enum):
    COCOA = 1
    COFFEE = 2
    COMPLEX_OIL_PALM = 3
    DENSE_FOREST = 4
    OIL_PALM = 5
    RUBBER = 6
    # UNKNOWN = 7
    SEASONAL_AGRICULTURE = 8
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
    BANANA = 19
    PALM_GROWING = 20
    CUT_OFF_REGROW = 21
    NATURAL_WETLAND = 22
    INTERCROP = 23
    DECIDUOUS_FOREST = 24
    STICK_FOR_PEPPER = 25
    FLOODED_PLANTATION = 26
    PINE_TREES = 27
    COCONUT = 28
    BAMBOO = 29
    SAVANA = 30
    MANGO = 31
    ORCHARD = 32
    MINE = 33
    SHRUBLAND_BUSHLAND = 34
    SPARE_TREE = 35
    BARE_SOIL = 36
    GRASSLAND = 37
    SECONDARY_DEGRADED_FOREST = 38
    MINE_BARESOIL = 39
    MINE_WATER = 40
    PEPPER_AND_COFFEE = 41
    PEPPER_AND_OTHER = 42
    SHADE_TREE = 43
    FARM_HOUSE = 44
    BURNED_LAND = 45
    ORCHARD_SMALL = 46
    LAKE = 47
    RIVER = 48
    COFFEE_FULL_SUN = 49
    COFFEE_SHADED = 50
    RUBBER_GROWING = 51
    CASHEW_AND_COCOA = 52
    CASHEW = 53
    ACACIA = 54
    ROAD = 55
    TEAK_PLANTATION_FOREST = 56
    PASTURE = 57
    PASSION_FRUIT = 58
    COCOA_SHADED = 59
    TRANSITION = 60
    LOTUS = 61
    GREENHOUSE = 62
    COFFEE_GROWING = 63
    MACADAMIA = 64
    MACADAMIA_GROWING = 65


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
    labels_coordinates = {label.value: [] for label in Label}

    # Add each points coordinate
    # in its corresponding class
    for path in shapefiles_paths:
        for shape_record in shapefile.Reader(path).shapeRecords():
            if shape_record.record.Class in labels_coordinates:
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
            label == Label.STICK_FOR_PEPPER or
            label == Label.SEASONAL_AGRICULTURE
    ):
        return LabelCategory.CULTURE
    elif (
            label == Label.DENSE_FOREST or
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


