from enum import Enum
import numpy as np


class Band(Enum):
    COASTAL_AEROSOL = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    NIR = 4
    SWIR1 = 5
    SWIR2 = 6
    TIRS1 = 7
    TIRS2 = 8
    NDVI = 9
    MNDWI = 10
    EVI2 = 11
    BU = 12


def add_ndvi_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        ndvi = (nir - red) / (nir + red)
        ndvi = (ndvi + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(ndvi.tolist())


def add_bu_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        swir = np.asarray(img[1][5])
        ndbi = (swir - nir) / (swir + nir)
        ndvi = (nir - red) / (nir + red)
        bu = ndbi - ndvi
        bu = (bu + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(bu.tolist())


def add_mndwi_to_dataset(dataset):
    for i, img in enumerate(dataset):
        green = np.asarray(img[1][2])
        swir = np.asarray(img[1][5])
        mndwi = (green - swir) / (green + swir)
        mndwi = (mndwi + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(mndwi.tolist())


def add_evi2_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        evi2 = 2.4 * (nir - red) / (nir + red + 1)
        evi2 = (evi2 + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(evi2.tolist())