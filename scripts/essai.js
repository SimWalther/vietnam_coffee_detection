// See: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_SR#bands
function maskL8sr(image) {
    // Bits 3 and 5 are cloud shadow and cloud, respectively.
    var cloudShadowBitMask = (1 << 3);
    var cloudsBitMask = (1 << 5);
  
    // Get the pixel QA band.
    var qa = image.select('pixel_qa');
  
    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  
    return image.updateMask(mask);
}

// Initial date (inclusive)
// Min start date is '2013-04-11'
var start_date = '2014-01-14';

// Final date (exclusive)
// Max end date is '2021-01-22'
var end_date = '2014-01-31';

var region = ee.Geometry.Rectangle([106.9998606274592134, 10.9999604855719539, 109.0000494390797456, 15.5002505644255208]);

// Get Vietnam geometry
var worldcountries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');
var filterCountry = ee.Filter.eq('country_na', 'Vietnam');
var country = worldcountries.filter(filterCountry);
var country_region = country.geometry();

// Select Landsat 8 Surface Reflectance Tier 1 dataset
var dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .map(maskL8sr)

// Reduce collection by median
var landsat8_img = dataset.median();

// Clip image to keep only the region of interest that is inside Vietnam
var landsat8_img_clipped = landsat8_img.clip(region).clip(country_region);

var visParams = {
  bands: ['B5', 'B4', 'B3'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};

Map.addLayer(landsat8_img_clipped, visParams);

