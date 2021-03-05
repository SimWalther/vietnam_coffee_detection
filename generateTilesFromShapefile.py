import shapefile
import glob
import rasterio.mask
import rasterio.warp
import shapely.geometry

from enum import Enum

NB_PIXEL_AROUND_POINT = 1


class ShapeClass(Enum):
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


def polygon_nb_px_around_coordinates(coordinate, n, dataset):
    lat = coordinate[0]
    lon = coordinate[1]

    row, col = dataset.index(lat, lon)

    min_coord = rasterio.transform.xy(dataset.transform, row - n, col - n, offset='ul')
    max_coord = rasterio.transform.xy(dataset.transform, row + n, col + n, offset='lr')

    return shapely.geometry.box(min_coord[0], min_coord[1], max_coord[0], max_coord[1])

# Open shapefiles
shapefile_root_path = "labels/"
shapefiles_paths = glob.glob(shapefile_root_path + "**/*.shp")

# Create a dictionary which will contain all
# points classified in the opened shapefiles
# with their coordinates lat-lon
classes_points_coordinates = {}

# Initialize the dictionary
for i in range(1, 34):
    classes_points_coordinates[i] = []

# Add each points coordinate
# in its corresponding class
for path in shapefiles_paths:
    sf = shapefile.Reader(path)
    print("\npath: " + path)
    print("----------------------------------------")

    for shape_record in sf.shapeRecords():
        if shape_record.record.Class <= 33:
            current_list = classes_points_coordinates.get(shape_record.record.Class)
            current_list.append(shape_record.shape.points[0])
            classes_points_coordinates[shape_record.record.Class] = current_list

# Open raster
RASTER_PATH = './merged.tif'
original_map = rasterio.open(RASTER_PATH)
original_metadata = original_map.meta
original_crs = original_map.crs.to_proj4()

shape_class = ShapeClass.COFFEE.value

i = 0
for coordinates in classes_points_coordinates[shape_class]:
    polygon = polygon_nb_px_around_coordinates(coordinates, NB_PIXEL_AROUND_POINT, original_map)
    out_img, out_transform = rasterio.mask.mask(original_map, shapes=[polygon], crop=True)

    # Write this tile to disk
    out_meta = original_metadata.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": original_crs
    })

    # Write merged raster to disk
    with rasterio.open('./tiles/' + str(ShapeClass(shape_class).name) + '_' + str(i) + '.tif', "w", **out_meta) as dest:
        dest.write(out_img)

    red = out_img[3]   # 4 - 1
    green = out_img[2] # 3 - 1
    blue = out_img[1]  # 2 - 1

    i += 1

    break
