import os

DATA_ROOT_PATH = os.path.abspath("../data/")
DATASET_PATH = os.path.join(DATA_ROOT_PATH, 'datasets/')
DATASET_IMAGES_PATH = os.path.join(DATA_ROOT_PATH, DATASET_PATH, 'images/')
RASTER_CHUNKS_PATH = os.path.join(DATA_ROOT_PATH, 'raster_chunks/')

MODEL_ROOT_PATH = os.path.abspath('../models/')
DISTRICTS_PATH = os.path.join(DATA_ROOT_PATH, 'districts', 'diaphantinh.geojson')
SOILMAP_PATH = os.path.join(DATA_ROOT_PATH, 'soilmap', 'soilmap.geojson')
SHAPEFILE_ROOT_PATH = os.path.join(DATA_ROOT_PATH, 'labels/')
