import geopandas as gpd
import shapely.geometry
import glob
from config import *
from labelsUtils import labels_coordinates_from_files

CENTRAL_HIGHLANDS_SHP_PATHS = glob.glob(os.path.join(SHAPEFILE_ROOT_PATH, 'central_highlands_*/*.shp'))
SUMATRA_SOUTH_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'sumatra_south/sumatra_south.shp')
SUMATRA_CENTER_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'sumatra_center/sumatra_center.shp')
SULAWESI_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'sulawesi/sulawesi.shp')
PARA_NORTH_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'para_north/para_north.shp')
PARA_CENTER_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'para_center/para_center.shp')
OCOTOPEQUE_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'ocotopeque/ocotopeque.shp')
GHANA_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'ghana/ghana.shp')
BORNEO_SHP_PATH = os.path.join(SHAPEFILE_ROOT_PATH, 'borneo/borneo.shp')

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


def sumatra_south_labels_coordinates():
    return labels_coordinates_from_files(
        [SUMATRA_SOUTH_SHP_PATH],
        shapely.geometry.box(103.335512, -3.659900, 105.97766784, -5.96634052)
    )


def sumatra_center_labels_coordinates():
    return labels_coordinates_from_files(
        [SUMATRA_CENTER_SHP_PATH],
        shapely.geometry.box(100.9018, 0.8373, 102.2407, -0.6134)
    )


def sulawesi_labels_coordinates():
    return labels_coordinates_from_files(
        [SULAWESI_SHP_PATH],
        shapely.geometry.box(118.6873, 1.3155, 123.137, -5.573)
    )


def para_north_labels_coordinates():
    return labels_coordinates_from_files(
        [PARA_NORTH_SHP_PATH],
        shapely.geometry.box(-55.0466, -2.6253, -52.5674, -4.4163)
    )


def para_center_labels_coordinates():
    return labels_coordinates_from_files(
        [PARA_CENTER_SHP_PATH],
        shapely.geometry.box(-53.4912, -5.4250, -50.9320, -7.1264)
    )


def ocotopeque_labels_coordinates():
    return labels_coordinates_from_files(
        [OCOTOPEQUE_SHP_PATH],
        shapely.geometry.box(-89.4848, 15.5488, -87.2884, 13.9015)
    )


def ghana_labels_coordinates():
    return labels_coordinates_from_files(
        [GHANA_SHP_PATH],
        shapely.geometry.box(-2.5215, 7.0745, -1.8946, 6.2379)
    )


def borneo_labels_coordinates():
    return labels_coordinates_from_files(
        [BORNEO_SHP_PATH],
        shapely.geometry.box(112.16164, -2.29216, 112.52132, -2.56859)
    )


def vietnam_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the region in Vietnam where we have labels.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    vietnam_shape = shapely.ops.unary_union([shape for shape in district_file['geometry']])
    selected_region = shapely.geometry.box(106.9998606274592134, 10.9999604855719539, 109.0000494390797456, 15.5002505644255208)
    boundaries_shape = vietnam_shape & selected_region
    return labels_coordinates_from_files(CENTRAL_HIGHLANDS_SHP_PATHS, boundaries_shape)


def highland_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the highland region in Vietnam.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    highland_districts_shape = shapely.ops.unary_union(shapes_from_geojson(district_file, HIGHLAND_DISTRICTS))
    return labels_coordinates_from_files(CENTRAL_HIGHLANDS_SHP_PATHS, highland_districts_shape)


def southern_labels_coordinates():
    """
    Get coordinates of each entries of every labels, takes only the southern region in Vietnam.
    :return: the labels coordinates dictionary
    """

    district_file = gpd.read_file(DISTRICTS_PATH)
    southern_districts_shape = shapely.ops.unary_union(shapes_from_geojson(district_file, SOUTHERN_DISTRICTS))
    return labels_coordinates_from_files(CENTRAL_HIGHLANDS_SHP_PATHS, southern_districts_shape)


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

    return [labels_coordinates_from_files(CENTRAL_HIGHLANDS_SHP_PATHS, soil) for soil in soils]


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

    return [labels_coordinates_from_files(CENTRAL_HIGHLANDS_SHP_PATHS, district) for district in districts]
