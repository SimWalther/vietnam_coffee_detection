import ee
import time
from datetime import datetime

# Initialize Google Earth Engine library
ee.Initialize()

# Landsat 8 collection
SATELLITE_DATASET = "LANDSAT/LC08/C01/T1_SR"

# Rectangle region of interest
REGION_RECTANGLE = ee.Geometry.Rectangle([
    106.9998606274592134, 10.9999604855719539,
    109.0000494390797456, 15.5002505644255208
])

# Country name to retrieve country geometry
COUNTRY_NAME = 'Vietnam'


def country_geometry(country_name):
    world_countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    filter_country = ee.Filter.eq('country_na', country_name)
    country = world_countries.filter(filter_country)

    return country.geometry()


def mask_clouds(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 5)

    # Get the pixel QA band.
    qa = image.select('pixel_qa')

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) and (qa.bitwiseAnd(clouds_bit_mask).eq(0))

    return image.updateMask(mask)


def start_task(task):
    task.start()
    print("Download task started...")

    while True:
        status = task.status()
        print(status)

        if status['state'] == 'COMPLETED':
            break
        elif status['state'] == 'FAILED':
            print(status['error_message'])
            break
        elif status['state'] == 'CANCEL_REQUESTED':
            break

        time.sleep(30)


# start date (inclusive)
# Min start date is '2013-04-11'
# end date (exclusive)
# Max end date is '2021-01-22'
def create_image_collection(start_date, end_date):
    # Select Landsat 8 dataset
    return ee.ImageCollection(SATELLITE_DATASET) \
        .filterDate(start_date, end_date) \
        .filterBounds(REGION_RECTANGLE) \
        .map(mask_clouds)


def create_export_task(img, folder_name):
    # Create an export to drive task
    return ee.batch.Export.image.toDrive(img, 'Vietnam', **{
        'folder': folder_name,
        'scale': 30,
        'maxPixels': 200_000_000,
        'fileFormat': 'GeoTIFF',
        'region': REGION_RECTANGLE
    })


def image_collection_to_median_img(image_collection):
    country_region = country_geometry(COUNTRY_NAME)

    # Reduce collection by median
    # and Clip image to keep only the region of interest that is inside the country
    return image_collection.median().clip(REGION_RECTANGLE).clip(country_region)


image_collection_2014 = create_image_collection('2014-06-01', '2014-12-01')
image_collection_2015 = create_image_collection('2015-06-01', '2015-12-01')
image_collection_2016 = create_image_collection('2016-06-01', '2016-12-01')
merged_image_collections = image_collection_2014 \
    .merge(image_collection_2015) \
    .merge(image_collection_2016) \

median_img = image_collection_to_median_img(merged_image_collections)
export_task = create_export_task(median_img, COUNTRY_NAME + '_' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))

# Start the task
start_task(export_task)
