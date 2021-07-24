import rasterio
import shapely.geometry
import rasterio.mask
import numpy as np
import geojson
from config import *


def image_around_coordinates(raster, coordinates, nb_pixel_around):
    """
    Retrieves raster image n pixels around a given latitude and longitude
    :param raster: the raster from which the data will be retrieved
    :param coordinates: the latitude and longitude tuple
    :param nb_pixel_around: the number of pixel around the coordinates
    For example: 4px means that it will get a square of 9x9 px. (4px + 1px + 4px)^2
    :return: the raster image, the affine transform
    """

    polygon = square_around_coordinates(raster, coordinates, nb_pixel_around)
    return rasterio.mask.mask(raster, shapes=[polygon], crop=True, all_touched=True)


def square_around_coordinates(raster, coordinate, nb_pixel_around):
    """
    Create a square of a given number of pixels around a latitude and longitude tuple.
    :param raster: the raster from which the data will be retrieved
    :param coordinate: the latitude and longitude tuple
    :param nb_pixel_around: the number of pixels around a given (lat, lon)
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


def image_square_at_px(row, col, square_size, raster):
    """
    Create a square image of a given number of pixels around a px row-col tuple.
    Given row col will be the upper left corner pixels row and column.
    :param raster: the raster from which the data will be retrieved
    :param row: the px row
    :param col: the px col
    :param square_size: the square size in pixels
    :return: a geojson object of the square
    """
    min_coord = rasterio.transform.xy(raster.transform, row, col)
    max_coord = rasterio.transform.xy(raster.transform, row + (square_size - 1), col + (square_size- 1))
    square = shapely.geometry.box(min_coord[0], min_coord[1], max_coord[0], max_coord[1])
    return rasterio.mask.mask(raster, shapes=[square], crop=True, all_touched=True)


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
        out_img, out_transform = image_around_coordinates(raster, coordinates, nb_pixel_around)

        # Don't add data if all values are 'NaN'
        if not np.isnan(out_img).all():
            # Add median values of each band (tirs1, tirs2) while filtering nodata values
            values.append([np.nanmedian(out_img[i]) for i in range(0, 7)])

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


def create_image_metadata(out_img, out_transform, original_raster):
    """
    Create image metadata by taking raster metadata as a basis and updating them.
    :param out_img: the raster image
    :param out_transform: the raster affine transform
    :param original_raster: the original raster
    :return: created metadata
    """
    metadata = original_raster.meta.copy()
    metadata.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": original_raster.crs
    })

    return metadata


def make_dataset_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around, save_on_disk=False, dataset_folder_name=""):
    """
    Make a dataset by taking images around provided labels. Multiple rasters can be given and in this case images are
    taken inside every rasters.
    Dataset will not include images with 'NaN' values.
    :param labels: the labels to use
    :param raster_paths: the paths to rasters from which to take the values
    :param labels_coordinates_list: a list of dictionary of all coordinates for each labels.
    This a list to enable having multiple dictionaries created separately without having to merge them into one
    :param nb_pixel_around: the number of pixel to take around labels
    :param save_on_disk: defines if images need to be saved on disk
    :param dataset_folder_name: name of the folder where to save images to
    :return: the dataset
    """

    values = []

    for i, path in enumerate(raster_paths):
        with rasterio.open(path) as raster:
            for labels_coordinates in labels_coordinates_list:
                for label in labels:
                    folder_path = None

                    if save_on_disk:
                        folder_path = os.path.join(DATASET_IMAGES_PATH, dataset_folder_name, label.name.lower())

                        # Create directories if they don't exists
                        os.makedirs(folder_path, exist_ok=True)

                    for label_image_index, coordinates in enumerate(labels_coordinates[label.value]):
                        out_img, out_transform = image_around_coordinates(
                            raster, coordinates, nb_pixel_around
                        )

                        lat = coordinates[0]
                        lon = coordinates[1]
                        point = geojson.Point((lat, lon))

                        # Don't add data if there is 'NaN' values
                        if not np.isnan(out_img).any():
                            if save_on_disk:
                                filepath = os.path.join(folder_path, str(label_image_index) + '.tiff')

                                # append filepath in dataset where raster is saved not the whole raster
                                values.append([label.name, filepath, point])

                                metadata = create_image_metadata(out_img, out_transform, raster)

                                # Write merged raster to disk
                                with rasterio.open(filepath, "w", **metadata) as dest:
                                    dest.write(out_img)
                            else:
                                # append whole rasters to dataset if we don't save rasters on the disk
                                values.append([label.name, out_img.tolist(), point])

    return values


def square_chunks(raster_path, square_size, batch_size=32):
    """
    Split raster into chunks of a specified size
    :param raster_path: the raster path
    :param square_size:
    :param batch_size:
    :return:
    """
    with rasterio.open(raster_path) as raster:
        width = raster.shape[0]
        height = raster.shape[1]
        nb_images_row = width // square_size
        nb_images_col = height // square_size

        print(f"Image width: {width}")
        print(f"Image height: {height}")
        print(f"Nb row of images: {nb_images_row}")
        print(f"Nb col of images: {nb_images_col}")

        if width % square_size != 0:
            print(f"Width is not dividable by {square_size}, some px will be ignored...")

        if height % square_size != 0:
            print(f"Height is not dividable by {square_size}, some px will be ignored...")

        img_count = 0
        batch_images = []
        batch_images_indices = []

        for i in range(nb_images_row):
            for j in range(nb_images_col):
                out_img, out_transform = image_square_at_px(i * square_size, j * square_size, square_size, raster)

                # Don't add if there is 'NaN' values
                if not np.isnan(out_img).any():
                    batch_images.append(out_img)
                    batch_images_indices.append((i, j))
                    img_count += 1

                if img_count == batch_size:
                    yield batch_images, batch_images_indices
                    # reset batch
                    batch_images = []
                    batch_images_indices = []
                    img_count = 0

        # if there is a leftover of images (less remaining images than the batch size)
        # yield them
        if len(batch_images) > 0:
            yield batch_images, batch_images_indices
