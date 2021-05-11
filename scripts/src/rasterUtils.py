import rasterio
import shapely.geometry
import rasterio.mask
import numpy as np
import geojson

DATA_ROOT_PATH = '../data/'
TILES_PATH = DATA_ROOT_PATH + 'tiles/'


def raster_n_px_around_tagged_point(coordinates, nb_pixel_around, initial_raster):
    polygon = polygon_nb_px_around_coordinates(coordinates, nb_pixel_around, initial_raster)
    return rasterio.mask.mask(initial_raster, shapes=[polygon], crop=True, all_touched=True)


def polygon_nb_px_around_coordinates(coordinate, n, dataset):
    lat = coordinate[0]
    lon = coordinate[1]

    row, col = dataset.index(lat, lon, precision=23)

    min_coord = rasterio.transform.xy(dataset.transform, row - n, col - n)
    max_coord = rasterio.transform.xy(dataset.transform, row + n, col + n)

    return shapely.geometry.box(min_coord[0], min_coord[1], max_coord[0], max_coord[1])


def labels_points_values(label, initial_raster, labels_coordinates, nb_pixel_around):
    values = []

    for coordinates in labels_coordinates[label.value]:
        out_img, out_transform = raster_n_px_around_tagged_point(coordinates, nb_pixel_around, initial_raster)

        # Don't add data if all values are 'NaN'
        if not np.isnan(out_img).all():
            # Add median values of each band (excepted cirrus, tirs1, tirs2) while filtering nodata values
            values.append([np.nanmedian(out_img[i]) for i in range(0, 8)])

    return np.array(values)


def labels_values_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around):
    values = []

    for path in raster_paths:
        with rasterio.open(path) as raster:
            for labels_coordinates in labels_coordinates_list:
                for label in labels:
                    values.append(labels_points_values(label, raster, labels_coordinates, nb_pixel_around))

    return values


def make_dataset_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around):
    values = []

    for i, path in enumerate(raster_paths):
        with rasterio.open(path) as raster:
            for labels_coordinates in labels_coordinates_list:
                for label in labels:
                    for coordinates in labels_coordinates[label.value]:
                        out_img, out_transform = raster_n_px_around_tagged_point(
                            coordinates, nb_pixel_around, raster
                        )

                        lat = coordinates[0]
                        lon = coordinates[1]
                        point = geojson.Point((lat, lon))

                        # Don't add data if there is 'NaN' values
                        if not np.isnan(out_img).any():
                            values.append([label.name, out_img.tolist(), point])

    return values
