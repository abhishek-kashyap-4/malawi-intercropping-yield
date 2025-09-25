# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:43:12 2025

@author: kashy
"""

import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import re

def to_region(path , shapefile = -1):
    src = rio.open(path)
    #gdf = gpd.read_file(r'Admin_Zones\Wolayita\wolayita.shp') 
    if(shapefile == -1):
        gdf = gpd.read_file(r'Admin_Zones\Wolayita\wolayita.shp')    
    else:
        gdf = gpd.read_file(shapefile) 
    with rio.open(path) as src:  # Replace with your TIFF path
        # Get the geometries from the GeoDataFrame
        geometries = gdf.geometry.values
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "count": 1,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    src = None
    # Save the clipped raster to a new file
    with rio.open(path, "w", **out_meta) as dest:
        dest.write(out_image)
        




def filter_belg(var , feature = -1):
    doy_start = 32 
    doy_end = 150
    if(feature == 'cpc_precip'):
        regex = r'_\d{7}_'
        mach = re.search(regex , var) 
        if(mach):
            doy = int(mach.group().strip('_')[4:])
            if(doy>= doy_start and doy<=doy_end):
                print(1)
                return 1 
            else:
                print(0)
                return 0
        else:
            raise Exception
        
    else:
        print(-1)
        return -1 