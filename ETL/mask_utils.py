# taken from https://gist.github.com/avanetten
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
    Create building mask for rasterSrc based upon vectorSrc. 
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

def plot_building_mask(input_image, pixel_coords, mask_image,   
                  figsize=(8,8), plot_name='',
                  add_title=False, poly_face_color='orange', 
                  poly_edge_color='red', poly_nofill_color='blue', cmap='bwr'):


    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, 
                                        figsize=(3*figsize[0], figsize[1]))
    
    if add_title:
        suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')

    # create patches
    patches = []
    patches_nofill = []
    if len(pixel_coords) > 0:
        # get patches    
        for coord in pixel_coords:
            patches_nofill.append(Polygon(coord, facecolor=poly_nofill_color, 
                                          edgecolor=poly_edge_color, lw=3))
            patches.append(Polygon(coord, edgecolor=poly_edge_color, fill=True, 
                                   facecolor=poly_face_color))
        p0 = PatchCollection(patches, alpha=0.25, match_original=True)
        p1 = PatchCollection(patches_nofill, alpha=0.75, match_original=True)
        
    #if len(patches) > 0:
    #    p0 = PatchCollection(patches, alpha=0.25, match_original=True)
    #    #p1 = PatchCollection(patches, alpha=0.75, match_original=True)
    #    p1 = PatchCollection(patches_nofill, alpha=0.75, match_original=True)                   
 
    # ax0: raw image
    ax0.imshow(input_image)
    if len(patches) > 0:
        ax0.add_collection(p0)
    ax0.set_title('Input Image + Ground Truth Buildings') 
    
    # truth polygons
    zero_arr = np.zeros(input_image.shape[:2])
    # set background to white?
    #zero_arr[zero_arr == 0.0] = np.nan
    ax1.imshow(zero_arr, cmap=cmap)
    if len(patches) > 0:
        ax1.add_collection(p1)
    ax1.set_title('Ground Truth Building Polygons')
        
    # old method of truth, with mask
    ## ax0: raw imageø
    #ax0.imshow(input_image)
    ## ground truth
    ## set zeros to nan
    #palette = plt.cm.gray
    #palette.set_over('orange', 1.0)
    #z = mask_image.astype(float)
    #z[z==0] = np.nan
    #ax0.imshow(z, cmap=palette, alpha=0.25, 
    #        norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9, clip=False))
    #ax0.set_title('Input Image + Ground Truth Buildings') 
   
    # mask
    ax2.imshow(mask_image, cmap=cmap)
    # truth polygons?
    #if len(patches) > 0:
    #    ax1.add_collection(p1)
    ax2.set_title('Ground Truth Building Mask')    
          
    #plt.axis('off')
    plt.tight_layout()
    if add_title:
        suptitle.set_y(0.95)
        fig.subplots_adjust(top=0.96)
    #plt.show()
 
    if len(plot_name) > 0:
        plt.savefig(plot_name)
    
    return

def plot_truth_coords(input_image, pixel_coords,   
                  figsize=(8,8), plot_name='',
                  add_title=False, poly_face_color='orange', 
                  poly_edge_color='red', poly_nofill_color='blue', cmap='bwr'):
    '''Plot ground truth coordinaates, pixel_coords should be a numpy array'''
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))
    
    if add_title:
        suptitle = fig.suptitle(plot_name.split('/')[-1], fontsize='large')
    
    # create patches
    patches = []
    patches_nofill = []
    if len(pixel_coords) > 0:
        # get patches    
        for coord in pixel_coords:
            patches_nofill.append(Polygon(coord, facecolor=poly_nofill_color, 
                                          edgecolor=poly_edge_color, lw=3))
            patches.append(Polygon(coord, edgecolor=poly_edge_color, fill=True, 
                                   facecolor=poly_face_color))
        p0 = PatchCollection(patches, alpha=0.25, match_original=True)
        #p1 = PatchCollection(patches, alpha=0.75, match_original=True)
        p2 = PatchCollection(patches_nofill, alpha=0.75, match_original=True)
                   
    # ax0: raw image
    ax0.imshow(input_image)
    if len(patches) > 0:
        ax0.add_collection(p0)
    ax0.set_title('Input Image + Ground Truth Buildings') 
    
    # truth polygons
    zero_arr = np.zeros(input_image.shape[:2])
    # set background to white?
    #zero_arr[zero_arr == 0.0] = np.nan
    ax1.imshow(zero_arr, cmap=cmap)
    if len(patches) > 0:
        ax1.add_collection(p2)
    ax1.set_title('Ground Truth Building Polygons')
        
    #plt.axis('off')
    plt.tight_layout()
    if add_title:
        suptitle.set_y(0.95)
        fig.subplots_adjust(top=0.96)
    #plt.show()
 
    if len(plot_name) > 0:
        plt.savefig(plot_name)
    
    return patches, patches_nofill
    

def geojson_to_pixel_arr(raster_file, geojson_file, pixel_ints=True,
                       verbose=False):
    '''
    Tranform geojson file into array of points in pixel (and latlon) coords
    pixel_ints = 1 sets pixel coords as integers
    '''
    
    # load geojson file
    with open(geojson_file) as f:
        geojson_data = json.load(f)

    # load raster file and get geo transforms
    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())
        
    geom_transform = src_raster.GetGeoTransform()
    
    # get latlon coords
    latlons = []
    types = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        type_tmp = feature['geometry']['type']
        if verbose: 
            print("features:", feature.keys())
            print("geometry:features:", feature['geometry'].keys())

            #print "feature['geometry']['coordinates'][0]", z
        latlons.append(coords_tmp)
        types.append(type_tmp)
        #print feature['geometry']['type']
    
    # convert latlons to pixel coords
    pixel_coords = []
    latlon_coords = []
    for i, (poly_type, poly0) in enumerate(zip(types, latlons)):
        
        if poly_type.upper() == 'MULTIPOLYGON':
            #print "oops, multipolygon"
            for poly in poly0:
                poly=np.array(poly)
                if verbose:
                    print("poly.shape:", poly.shape)
                    
                # account for nested arrays
                if len(poly.shape) == 3 and poly.shape[0] == 1:
                    poly = poly[0]
                    
                poly_list_pix = []
                poly_list_latlon = []
                if verbose: 
                    print("poly", poly)
                for coord in poly:
                    if verbose: 
                        print("coord:", coord)
                    lon, lat, z = coord 
                    px, py = latlon2pixel(lat, lon, input_raster=src_raster,
                                         targetsr=targetsr, 
                                         geom_transform=geom_transform)
                    poly_list_pix.append([px, py])
                    if verbose:
                        print("px, py", px, py)
                    poly_list_latlon.append([lat, lon])
                
                if pixel_ints:
                    ptmp = np.rint(poly_list_pix).astype(int)
                else:
                    ptmp = poly_list_pix
                pixel_coords.append(ptmp)
                latlon_coords.append(poly_list_latlon)            

        elif poly_type.upper() == 'POLYGON':
            poly=np.array(poly0)
            if verbose:
                print("poly.shape:", poly.shape)
                
            # account for nested arrays
            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]
                
            poly_list_pix = []
            poly_list_latlon = []
            if verbose: 
                print("poly", poly)
            for coord in poly:
                if verbose: 
                    print("coord:", coord)
                lon, lat, z = coord 
                px, py = latlon2pixel(lat, lon, input_raster=src_raster,
                                     targetsr=targetsr, 
                                     geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                if verbose:
                    print("px, py", px, py)
                poly_list_latlon.append([lat, lon])
            
            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)
            
        else:
            print("Unknown shape type in coords_arr_from_geojson()")
            return
            
    return pixel_coords, latlon_coords




# taken from https://github.com/SpaceNetChallenge/utilities/tree/master

def latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    # type: (object, object, object, object, object) -> object

    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(lon, lat)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)
    geom.Transform(coord_trans)
    # print(geom.GetPoint())
    x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
    y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height

    return (x_pix, y_pix)
