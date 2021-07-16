import glob

import rasterio
import rasterio.merge
import rasterio.plot
import time

ROOT_PATH = "../data/"

FOLDER_NAMES = [
    "Sumatra_south_2018_whole_year_collection2"
]

def open_raster_files(folder_path):
    """
    Open rasters inside a folder
    :param folder_path: the folder path
    :return: files opened by rasterio
    """
    raster_paths = glob.glob(ROOT_PATH + folder_path + "/*.tif")
    return list(map(rasterio.open, raster_paths))


# For each folder in FOLDER_NAMES, merge all rasters to a file named 'merged.tif'
for folder_name in FOLDER_NAMES:
    print("Merging raster files...")
    raster_files = open_raster_files(folder_name)
    merged_result, out_transform = rasterio.merge.merge(raster_files)
    print("Raster files merged")

    metadata = raster_files[0].meta.copy()
    metadata.update({
        "driver": "GTiff",
        "height": merged_result.shape[1],
        "width": merged_result.shape[2],
        "transform": out_transform,
        "crs": raster_files[0].crs
    })

    print("Writing merged raster to disk...")

    # Write merged raster to disk
    with rasterio.open(ROOT_PATH + folder_name + '/merged.tif', "w", **metadata) as dest:
        dest.write(merged_result)

    print("Finished writing to disk")

    time.sleep(45)  # sleep a little to let memory usage go down
