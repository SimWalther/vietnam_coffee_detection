import geopandas as gpd
import shapely.geometry
import glob
from labelsUtils import labels_coordinates_from_files

DATA_ROOT_PATH = '../data/'
DISTRICTS_PATH = DATA_ROOT_PATH + "districts/diaphantinh.geojson"
SOILMAP_PATH = DATA_ROOT_PATH + "soilmap/soilmap.geojson"
SHAPEFILE_ROOT_PATH = DATA_ROOT_PATH + 'labels/'
SHAPEFILE_PATHS = glob.glob(SHAPEFILE_ROOT_PATH + '**/*.shp')

HIGHLAND_DISTRICTS = [
    'Dak Nong',
    'Dak Lak',
    'Gia Lai',
    'Kon Tum'
]

SOUTHERN_DISTRICTS = [
    'Lam Dong',
    'Binh Phuoc',
    'Dong Nai',
    'Tay Ninh',
    'Binh Duong',
    'TP. Ho Chi Minh',
    'Ba Ria - Vung Tau',
    'Long An',
    'Tien Giang',
    'Ben Tre',
    'Dong Thap',
    'Vinh Long',
    'Tra Vinh',
    'An Giang',
    'Can Tho',
    'Hau Giang',
    'Soc Trang',
    'Kien Giang',
    'Bac Lieu',
    'Ca Mau'
]


def shapes_from_geojson(file, names, name_column='Name', geometry_column='geometry'):
    """
    Extract shapes from a geojson file
    :param file: the geojson file
    :param names: names to keep
    :param name_column: name of the column where the names are
    :param geometry_column: name of the column with the shape
    :return: the shapes
    """

    return [shape for shape in file[file[name_column].isin(names)][geometry_column]]


def vietnam_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the region in Vietnam where we have labels.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    vietnam_shape = shapely.ops.unary_union([shape for shape in district_file['geometry']])
    selected_region = shapely.geometry.box(106.9998606274592134, 10.9999604855719539, 109.0000494390797456, 15.5002505644255208)
    boundaries_shape = vietnam_shape & selected_region
    return labels_coordinates_from_files(SHAPEFILE_PATHS, boundaries_shape)


def highland_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the highland region in Vietnam.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    highland_districts_shape = shapely.ops.unary_union(shapes_from_geojson(district_file, HIGHLAND_DISTRICTS))
    return labels_coordinates_from_files(SHAPEFILE_PATHS, highland_districts_shape)


def southern_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the southern region in Vietnam.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    southern_districts_shape = shapely.ops.unary_union(shapes_from_geojson(district_file, SOUTHERN_DISTRICTS))
    return labels_coordinates_from_files(SHAPEFILE_PATHS, southern_districts_shape)


def soils_labels_coordinates():
    """
    Get coordinates of each entries of every labels, create one labels dictionary by soil type.
    :return: array of labels coordinates dictionaries
    """

    soil_file = gpd.read_file(SOILMAP_PATH)

    soils = [
        shapely.ops.unary_union(shapes_from_geojson(soil_file, [soil], 'domsoil'))
        for soil
        in ['Fa', 'Af', 'Ao', 'Fr', 'Fo']
    ]

    return [labels_coordinates_from_files(SHAPEFILE_PATHS, soil) for soil in soils]


def districts_labels_coordinates():
    """
    Get coordinates of each entries of every labels, create one labels dictionary by districts.
    :return: array of labels coordinates dictionaries
    """

    district_file = gpd.read_file(DISTRICTS_PATH)

    districts = [
        shapely.ops.unary_union(shapes_from_geojson(district_file, [district]))
        for district
        in ['Gia Lai', 'Dak Lak', 'Dak Nong', 'Lam Dong']
    ]

    return [labels_coordinates_from_files(SHAPEFILE_PATHS, district) for district in districts]
