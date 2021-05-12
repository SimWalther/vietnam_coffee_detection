import ee
import time

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
    """
    Get a country geometry
    :param country_name: the country name
    :return: the country geometry
    """

    world_countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    filter_country = ee.Filter.eq('country_na', country_name)
    country = world_countries.filter(filter_country)

    return country.geometry()


def mask_clouds(image):
    """
    mask clouds in a landsat 8 image.
    See: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR?hl=in&skip_cache=false
    :param image: the image
    :return: image without clouds
    """

    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 5)

    # Get the pixel QA band.
    qa = image.select('pixel_qa')

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) and (qa.bitwiseAnd(clouds_bit_mask).eq(0))

    return image.updateMask(mask)


def start_task(task):
    """
    Start a google earth engine task while keeping tracks of its state
    :param task: the task
    """

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


def create_image_collection(start_date, end_date):
    """
    Create an image collection containing satellite image in REGION_RECTANGLE and within the time range.
    Warning: the end_date is excluded
    Min start date is '2013-04-11'
    Max end date is '2021-01-22'
    :param start_date: the start date
    :param end_date: the end date
    :return: the image collection
    """

    # Select Landsat 8 dataset
    return ee.ImageCollection(SATELLITE_DATASET) \
        .filterDate(start_date, end_date) \
        .filterBounds(REGION_RECTANGLE) \
        .map(mask_clouds)


def create_export_task(img, folder_name):
    """
    Create an export to drive task in Google Earth Engine
    :param img: the image to export
    :param folder_name: folder name in drive to export to
    :return: the task
    """

    return ee.batch.Export.image.toDrive(img, 'Vietnam', **{
        'folder': folder_name,
        'scale': 30,
        'maxPixels': 200_000_000,
        'fileFormat': 'GeoTIFF',
        'region': REGION_RECTANGLE
    })


def image_collection_to_median_img(image_collection):
    """
    Reduce an image collection to a median image and
    keep only parts of the image within REGION_RECTANGLE and the country shape
    :param image_collection:
    :return: the median image
    """
    country_region = country_geometry(COUNTRY_NAME)

    # Reduce collection by median
    # and Clip image to keep only the region of interest that is inside the country
    return image_collection.median().clip(REGION_RECTANGLE).clip(country_region)


# -------------------------- DRY SEASON
# image_collection_2016 = create_image_collection('2016-11-01', '2017-05-01')
# image_collection_2017 = create_image_collection('2017-11-01', '2018-05-01')
# image_collection_2018 = create_image_collection('2018-11-01', '2019-05-01')

# -------------------------- WET SEASON
# image_collection_2016 = create_image_collection('2016-05-01', '2016-10-01')
# image_collection_2017 = create_image_collection('2017-05-01', '2017-10-01')
# image_collection_2018 = create_image_collection('2018-05-01', '2018-10-01')
#
# merged_image_collections = image_collection_2016 \
#     .merge(image_collection_2017) \
#     .merge(image_collection_2018)

# -------------------------- 2 MONTHS
# image_collection_2014 = create_image_collection('2014-01-01', '2014-03-01')
# image_collection_2015 = create_image_collection('2015-01-01', '2015-03-01')
# image_collection_2016 = create_image_collection('2016-01-01', '2016-03-01')
# image_collection_2017 = create_image_collection('2017-01-01', '2017-03-01')
# image_collection_2018 = create_image_collection('2018-01-01', '2018-03-01')
# image_collection_2019 = create_image_collection('2019-01-01', '2019-03-01')
# image_collection_2020 = create_image_collection('2020-01-01', '2020-03-01')

# image_collection_2014 = create_image_collection('2014-03-01', '2014-05-01')
# image_collection_2015 = create_image_collection('2015-03-01', '2015-05-01')
# image_collection_2016 = create_image_collection('2016-03-01', '2016-05-01')
# image_collection_2017 = create_image_collection('2017-03-01', '2017-05-01')
# image_collection_2018 = create_image_collection('2018-03-01', '2018-05-01')
# image_collection_2019 = create_image_collection('2019-03-01', '2019-05-01')
# image_collection_2020 = create_image_collection('2020-03-01', '2020-05-01')
#
# image_collection_2014 = create_image_collection('2014-05-01', '2014-07-01')
# image_collection_2015 = create_image_collection('2015-05-01', '2015-07-01')
# image_collection_2016 = create_image_collection('2016-05-01', '2016-07-01')
# image_collection_2017 = create_image_collection('2017-05-01', '2017-07-01')
# image_collection_2018 = create_image_collection('2018-05-01', '2018-07-01')
# image_collection_2019 = create_image_collection('2019-05-01', '2019-07-01')
# image_collection_2020 = create_image_collection('2020-05-01', '2020-07-01')
#
# image_collection_2014 = create_image_collection('2014-07-01', '2014-09-01')
# image_collection_2015 = create_image_collection('2015-07-01', '2015-09-01')
# image_collection_2016 = create_image_collection('2016-07-01', '2016-09-01')
# image_collection_2017 = create_image_collection('2017-07-01', '2017-09-01')
# image_collection_2018 = create_image_collection('2018-07-01', '2018-09-01')
# image_collection_2019 = create_image_collection('2019-07-01', '2019-09-01')
# image_collection_2020 = create_image_collection('2020-07-01', '2020-09-01')
#
# image_collection_2014 = create_image_collection('2014-09-01', '2014-11-01')
# image_collection_2015 = create_image_collection('2015-09-01', '2015-11-01')
# image_collection_2016 = create_image_collection('2016-09-01', '2016-11-01')
# image_collection_2017 = create_image_collection('2017-09-01', '2017-11-01')
# image_collection_2018 = create_image_collection('2018-09-01', '2018-11-01')
# image_collection_2019 = create_image_collection('2019-09-01', '2019-11-01')
# image_collection_2020 = create_image_collection('2020-09-01', '2020-11-01')
#
# image_collection_2014 = create_image_collection('2014-11-01', '2015-01-01')
# image_collection_2015 = create_image_collection('2015-11-01', '2016-01-01')
# image_collection_2016 = create_image_collection('2016-11-01', '2017-01-01')
# image_collection_2017 = create_image_collection('2017-11-01', '2018-01-01')
# image_collection_2018 = create_image_collection('2018-11-01', '2019-01-01')
# image_collection_2019 = create_image_collection('2019-11-01', '2020-01-01')
# image_collection_2020 = create_image_collection('2020-11-01', '2021-01-01')

# -------------------------- 2014 - 2020
# image_collection = create_image_collection('2014-01-01', '2021-01-01')

# -------------------------- January to april, 2017
# image_collection = create_image_collection('2017-01-01', '2017-05-01')

# -------------------------- January to March, 2017
# image_collection = create_image_collection('2017-01-01', '2017-03-01')

# -------------------------- March to April, 2017
image_collection = create_image_collection('2017-03-01', '2017-05-01')


median_img = image_collection_to_median_img(image_collection)
export_task = create_export_task(median_img, '2017_march_april')

# Start the task
start_task(export_task)
