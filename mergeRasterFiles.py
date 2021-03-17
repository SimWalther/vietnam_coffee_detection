import glob

import rasterio
import rasterio.merge
import rasterio.plot

FOLDER_NAMES = [
    "Vietnam_2014_2020"
]


def open_raster_files(folder_path):
    raster_paths = glob.glob(folder_path + "*.tif")
    return list(map(rasterio.open, raster_paths))


for folder_name in FOLDER_NAMES:
    folder_path = './' + folder_name + '/'

    print("Merging raster files...")
    raster_files = open_raster_files(folder_path)
    merged_result, out_transform = rasterio.merge.merge(raster_files)
    print("Raster files merged")

    metadata = raster_files[0].meta.copy()
    metadata.update({
        "driver": "GTiff",
        "height": merged_result.shape[1],
        "width": merged_result.shape[2],
        "transform": out_transform,
        "crs": raster_files[0].crs.to_proj4()
    })

    print("Writing merged raster to disk...")

    # Write merged raster to disk
    with rasterio.open(folder_path + 'merged.tif', "w", **metadata) as dest:
        dest.write(merged_result)

    print("Finished writing to disk")
