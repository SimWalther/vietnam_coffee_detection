import ee
import time


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


# Initialize Google Earth Engine library
ee.Initialize()

# Initial date (inclusive)
# Min start date is '2013-04-11'
start_date = '2014-01-14'

# Final date (exclusive)
# Max end date is '2021-01-22'
end_date = '2014-03-04'

# Set region of interest
region = ee.Geometry.Rectangle([106.9998606274592134, 10.9999604855719539, 109.0000494390797456, 15.5002505644255208])

# Get Vietnam geometry
vietnam_region = country_geometry('Vietnam')

# Landsat 8 collection
image_collection = "LANDSAT/LC08/C01/T1_SR"

# Select Landsat 8 dataset
dataset = ee.ImageCollection(image_collection) \
    .filterDate(start_date, end_date) \
    .filterBounds(region) \
    .map(mask_clouds)

# Reduce collection by median
landsat8_img = dataset.median()

# Clip image to keep only the region of interest that is inside Vietnam
landsat8_img_clipped = landsat8_img.clip(region).clip(vietnam_region)

# Create an export to drive task
export_task = ee.batch.Export.image.toDrive(landsat8_img_clipped, 'Vietnam', **{
    'folder': 'TB',
    'scale': 30,
    'maxPixels': 200_000_000,
    'fileFormat': 'GeoTIFF',
    'region': region
})

# Start the task
start_task(export_task)
