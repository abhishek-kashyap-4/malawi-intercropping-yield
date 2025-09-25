import geopandas as gpd
import rasterio
import rasterstats  
import pandas as pd
import os
from tqdm import tqdm



def get_agvar(files_path , shapefile_path ,agvar, agg = "mean"): 

    gdf = gpd.read_file(shapefile_path)
    if not all(gdf.geometry.geom_type == 'Point'):
        gdf['geometry'] = gdf.geometry.centroid

    tiff_files = sorted([f for f in os.listdir(files_path) if f.endswith('.tif')])
    #time_series = pd.DataFrame(index=gdf.index)
    time_series = {}
    for i, tiff_file in enumerate(tqdm(tiff_files)):
        
        tiff_path = os.path.join(files_path, tiff_file)

        with rasterio.open(tiff_path) as src: 
            if gdf.crs != src.crs:
                gdf.to_crs(src.crs) 
            coords = [(geom.x, geom.y) for geom in gdf.geometry]
            values = list(src.sample(coords))
            values  =  [val[0] for val in values]
            if(agvar == 'Precipitation'):
                append = tiff_file.split('_')[1].split('.')[-1]
            elif(agvar == 'Temperature'):
                append = tiff_file.split('_')[1]
            else: 
                2/0
            time_series[f'{agvar}__'+append] = values 

    time_series = pd.DataFrame(time_series)

    return time_series 





temperature_files_path = r"D:\Unistra\Malawi\AGROECO\Temperature"
Precipitation_files_path = r"D:\Unistra\Malawi\AGROECO\Precipitation"

shapefile_path = r"D:\Unistra\Malawi\AK\data\train_ready\malawi_triplet_eos\train_rise\harmonised_malawi_3857_revised_reproj\annotated.shp"
temp = get_agvar(temperature_files_path , shapefile_path,agvar = 'Temperature')
precip = get_agvar(Precipitation_files_path , shapefile_path,'Precipitation')
gdf = gpd.read_file(shapefile_path)
#gdf['geo_orig'] = gdf.geometry
gdf_out = gdf.join(temp).join(precip)


gdf_out.to_file('wip.csv') 