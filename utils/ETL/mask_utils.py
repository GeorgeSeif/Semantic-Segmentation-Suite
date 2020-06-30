# modified from https://gist.github.com/avanetten
# https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53



from osgeo import gdal, ogr, osr

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

import json
import sys



###############################################################################
def create_building_mask(rasterSrc, vectorSrc, npDistFileName='', 
                            output_channels=3, noDataValue=0, building_value=(255,0,0), perimeter_value=(0,0,255), perimeter_width=0):

    '''
    Creates building mask for rasterSrc based upon vectorSrc.
    Output dtype is uint8 (0,255)
    Will also draw building perimeter iff perimeter_wdith > 0.

        -rasterSrc: path to aerial image
        -vectorSrc: path to geojson
        -npDistFileName: path to output ground truth image
        -perimeter_width: width of the perimeter mask in approximate meters
    '''


    bands = list(range(1,output_channels +1))
    
    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    ## create First raster memory layer, units are pixels
    # Change output to geotiff instead of memory 
    memdrv = gdal.GetDriverByName('GTiff') 


    dst_ds = memdrv.Create(npDistFileName, cols, rows, output_channels, gdal.GDT_Byte, 
                           options=['COMPRESS=LZW'])

    #copy projection from raster
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())


    [dst_ds.GetRasterBand(i).SetNoDataValue(noDataValue) for i in bands]

    # draw perimeter mask before (underneath) the building mask
    if perimeter_width > 0:
        draw_building_perimeter(perimeter_width, bands, dst_ds, source_layer, perimeter_value)
        

    gdal.RasterizeLayer(dst_ds, bands, source_layer, burn_values=building_value)
    dst_ds = 0
    
    return 


def draw_building_perimeter(perimeter_width, bands, dst_ds, src_lyr, perimeter_value):

    # create temporary datasource in memory to hold buffered boundary
    temp_ds = ogr.GetDriverByName('Memory').CreateDataSource('wrk')
    temp_layer = temp_ds.CreateLayer('poly', srs = src_lyr.GetSpatialRef())
    temp_layer_def = temp_layer.GetLayerDefn()

    for feature in src_lyr:
        ingeom = feature.GetGeometryRef()

        base_buffer_dist = .00001 # measured in degrees- equiv. to 1.111 meters at the equator

        buffer_dist = perimeter_width * base_buffer_dist / 1.11 #convert degrees to meters

        geomBuffer = ingeom.Buffer(buffer_dist)

        temp_feat = ogr.Feature(temp_layer_def)
        temp_feat.SetGeometryDirectly(geomBuffer)

        temp_layer.CreateFeature(temp_feat)

    gdal.RasterizeLayer(dst_ds, bands, temp_layer, burn_values=perimeter_value)
    temp_ds = 0 # save file and flush memory

