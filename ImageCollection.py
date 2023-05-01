import os
import ee
import geemap

#We load the map from Google earth engine
Map = geemap.Map()
Map

#We define our are of interest on the map and we save it to region
region = Map.user_roi

#We use the imageCollection to extract more than 1 image and we define which days and bands we want to download.
collection  =  (ee.ImageCollection('COPERNICUS/S2')
           .filterBounds(region)
           .filter('CLOUDY_PIXEL_PERCENTAGE < 5')
           .filterDate('2023-01-12', '2023-04-20')
           .select('B12','B11','B4')
                )

# we choose color to visualize the image
color = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
               '74A901', '66A000', '529400', '3E8601', '207401', '056201',
               '004C00', '023B01', '012E01', '011D01', '011301']
pallete = {"min":0, "max":1, 'palette':color}



visparam = {
    'min': 0,
    'max': 6000,
    'gamma': 1.4,
}

Map.addLayer(collection, vis_params, "sentinel")
Map.setCenter(21.706095, 40.199855, 12)
Map

# Finally we download all the images that are met with the aforementioned criteria as .jpg
out_dir = os.path.expanduser("~/Desktop")
geemap.get_image_collection_thumbnails(collection,out_dir, vis_params,region=region)
