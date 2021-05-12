import rasterio
import shapely.geometry
import rasterio.mask
import numpy as np
import geojson


def image_n_px_around_coordinates(coordinates, nb_pixel_around, raster):
    """
    Retrieves raster image n pixels around a given latitude and longitude
    :param coordinates: the latitude and longitude tuple
    :param nb_pixel_around: the number of pixel around the coordinates
    For example: 4px means that it will get a square of 9x9 px. (4px + 1px + 4px)^2
    :param raster: the raster from which the data will be retrieved
    :return: the raster image, the affine transform
    """

    polygon = square_of_n_px_around_coordinates(coordinates, nb_pixel_around, raster)
    return rasterio.mask.mask(raster, shapes=[polygon], crop=True, all_touched=True)


def square_of_n_px_around_coordinates(coordinate, nb_pixel_around, raster):
    """
    Create a square of a given number of pixels around a latitude and longitude tuple.
    :param coordinate: the latitude and longitude tuple
    :param nb_pixel_around: the number of pixels around a given (lat, lon)
    :param raster: the raster from which the data will be retrieved
    :return: a geojson object of the square
    """

    lat = coordinate[0]
    lon = coordinate[1]

    # Latitude and longitude are converted to row and col of pixels in the raster
    row, col = raster.index(lat, lon, precision=23)

    # To get the min and max coordinates, n pixel are taken around the row and col of given coordinates and
    # then converted back to latitude and longitude.
    min_coord = rasterio.transform.xy(raster.transform, row - nb_pixel_around, col - nb_pixel_around)
    max_coord = rasterio.transform.xy(raster.transform, row + nb_pixel_around, col + nb_pixel_around)

    return shapely.geometry.box(min_coord[0], min_coord[1], max_coord[0], max_coord[1])


def label_median_values(label, raster, labels_coordinates, nb_pixel_around):
    """
    Get the median value of each label entry for each band (excepted cirrus, tirs1, tirs2)
    This function ignore images with only 'NaN' values and exclude 'Nan' values from the median computation
    :param label: the label
    :param raster: the raster from which the data will be retrieved
    :param labels_coordinates: a dictionary of all coordinates for each labels
    :param nb_pixel_around: the number of pixel around the label's coordinates
    :return: an array with the median values of each band for every label's entries
    """

    values = []

    for coordinates in labels_coordinates[label.value]:
        out_img, out_transform = image_n_px_around_coordinates(coordinates, nb_pixel_around, raster)

        # Don't add data if all values are 'NaN'
        if not np.isnan(out_img).all():
            # Add median values of each band (excepted cirrus, tirs1, tirs2) while filtering nodata values
            values.append([np.nanmedian(out_img[i]) for i in range(0, 8)])

    return np.array(values)


def labels_values_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around):
    """
    Get all median values of interesting bands for every labels and their entries.
    Those values are taken for every provided raster files.
    :param labels: the labels to use
    :param raster_paths: the paths to rasters from which to take the values
    :param labels_coordinates_list: a list of dictionary of all coordinates for each labels.
    This a list to enable having multiple dictionaries created separately without having to merge them into one
    :param nb_pixel_around: the number of pixel to take around labels
    :return: an array with all the computed values
    """

    values = []

    for path in raster_paths:
        with rasterio.open(path) as raster:
            for labels_coordinates in labels_coordinates_list:
                for label in labels:
                    values.append(label_median_values(label, raster, labels_coordinates, nb_pixel_around))

    return values


def make_dataset_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around):
    """
    Make a dataset by taking images around provided labels. Multiple rasters can be given and in this case images are
    taken inside every rasters.
    Dataset will not include images with 'NaN' values.
    :param labels: the labels to use
    :param raster_paths: the paths to rasters from which to take the values
    :param labels_coordinates_list: a list of dictionary of all coordinates for each labels.
    This a list to enable having multiple dictionaries created separately without having to merge them into one
    :param nb_pixel_around: the number of pixel to take around labels
    :return: the dataset
    """

    values = []

    for i, path in enumerate(raster_paths):
        with rasterio.open(path) as raster:
            for labels_coordinates in labels_coordinates_list:
                for label in labels:
                    for coordinates in labels_coordinates[label.value]:
                        out_img, out_transform = image_n_px_around_coordinates(
                            coordinates, nb_pixel_around, raster
                        )

                        lat = coordinates[0]
                        lon = coordinates[1]
                        point = geojson.Point((lat, lon))

                        # Don't add data if there is 'NaN' values
                        if not np.isnan(out_img).any():
                            values.append([label.name, out_img.tolist(), point])

    return values
