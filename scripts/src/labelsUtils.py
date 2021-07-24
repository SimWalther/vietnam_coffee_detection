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
    URBAN = 2
    WATER = 3
    OTHER_NATURE = 4
    OTHER_TREE = 5


CATEGORIES_LABELS = {
    LabelCategory.CULTURE: [
        Label.COCOA,
        Label.COFFEE,
        Label.RUBBER,
        Label.OIL_PALM,
        Label.CASSAVA,
        Label.SEASONAL_AGRICULTURE,
        Label.COMPLEX_OIL_PALM,
        Label.PEPPER,
        Label.TEA,
        Label.RICE,
        Label.BANANA,
        Label.PALM_GROWING,
        Label.INTERCROP,
        Label.BAMBOO,
        Label.COCONUT,
        Label.MANGO,
        Label.ORCHARD,
        Label.LOTUS,
        Label.CASHEW,
        Label.CASHEW_AND_COCOA,
        Label.COFFEE_GROWING,
        Label.COFFEE_SHADED,
        Label.COFFEE_FULL_SUN,
        Label.RUBBER_GROWING,
        Label.PASSION_FRUIT,
        Label.COCOA_SHADED,
        Label.MACADAMIA,
        Label.MACADAMIA_GROWING,
        Label.PEPPER_AND_COFFEE,
        Label.PEPPER_AND_OTHER,
        Label.FLOODED_PLANTATION,
        Label.ORCHARD_SMALL,
        Label.STICK_FOR_PEPPER
    ],
    LabelCategory.FOREST: [
        Label.SECONDARY_DEGRADED_FOREST,
        Label.DECIDUOUS_FOREST,
        Label.DENSE_FOREST
    ],
    LabelCategory.WATER: [
        Label.WATER,
        Label.WATER_OTHER,
        Label.LAKE,
        Label.RIVER
    ],
    LabelCategory.URBAN: [
        Label.URBAN,
        Label.ROAD,
        Label.FARM_HOUSE,
        Label.GREENHOUSE,
        Label.MINE,
        Label.MINE_BARESOIL,
        Label.MINE_WATER
    ],
    LabelCategory.OTHER_NATURE: [
        Label.PASTURE,
        Label.SAVANA,
        Label.OTHER_NO_TREE,
        Label.NATIVE_NO_TREE,
        Label.BURNED_LAND,
        Label.SHRUBLAND_BUSHLAND,
        Label.NATURAL_WETLAND,
        Label.CUT_OFF_REGROW,
        Label.BARE_SOIL,
        Label.GRASSLAND,
        Label.TRANSITION,
        Label.SPARE_TREE
    ],
    LabelCategory.OTHER_TREE: [
        Label.SHADE_TREE,
        Label.ACACIA,
        Label.TEAK_PLANTATION_FOREST,
        Label.PINE_TREES,
        Label.OTHER_TREE,
    ]
}


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
    """
    Get the category of a given label
    :param label: the label
    :return: the category
    """
    for labelCategory in LabelCategory:
        if label in CATEGORIES_LABELS[labelCategory]:
            return labelCategory

    return None


def categories_from_label_set(labels, images_labels):
    """
    Get the categories for a label set
    :param labels: the labels used
    :param images_labels: labels for each images
    :return: the categories for each images
    """
    return np.asarray([
        category_from_label(labels[label]).value for label in images_labels
    ])


