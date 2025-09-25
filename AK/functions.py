

#generic
from pathlib import Path
import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import warnings
#ML 

#RS 
import geopandas as gpd
import rasterio

#Project




def convert_to_df(data , bandvalues): 
    """
    Given the input in the form of data.npy , ids.npy , and meta gdf , 
    Convert the whole thing to gdf (pixels , timesteps*bands , metadata)
    Rename columns as timestep__band 
    Data is of the form - (pixels ,  timesteps , bands )

    """
    assert len(bandvalues) ==  data.shape[2]
    
    df = {}
    for i in range(len(bandvalues)): 
        band = data[:,:,i] 
        bandname = bandvalues[i] 
        for j in range(data.shape[1]): 
            df[f'{bandname}__{j}']  = band[:,j]
            
        pass 
    
    return pd.DataFrame(df)


def apply_field_reduction(data: np.ndarray, ids: np.ndarray , how:str = 'median') -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the median of samples grouped by unique field IDs.

        Args:
            data (np.ndarray): Input time-series data of shape (samples, time, bands).
            ids (np.ndarray): Array of field IDs of shape (samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Median data of shape (unique_fields, time, bands), unique field IDs.
        """
        unique_ids = np.unique(ids)
        reduced_data = np.zeros((len(unique_ids), data.shape[1], data.shape[2]), dtype=data.dtype)

        for i, uid in enumerate(unique_ids):
            sample_mask = np.where(ids == uid)[0]
            group_samples = data[sample_mask]
            if(how =='median'): 
                reduced_data[i] = np.nanmedian(group_samples, axis=0)
            elif(how == 'mean'): 
                 reduced_data[i] = np.nanmean(group_samples, axis=0)
            elif(how == 'std' ): 
                 reduced_data[i] = np.nanstd(group_samples , axis = 0 )
            else: 
                 raise NotImplementedError


        return reduced_data, unique_ids

def apply_aggregation(df  , kind = 'mean_max' ): 
     """
     This method has to be applied to the dataframe version of the code. 
     Given the df and method (kind) , perform the aggregation. 
     If you want aggregation to a sepcific band, you should filter it out before and only sent the subset.
     Kind is written such that you can multiple at the same time. 
     """
     kind = kind.split('_') # this should work even if underscore isn't in the thing. 
     d = {}
     if('mean' in kind): 
         d['mean']  = df.mean(axis=1 , numeric_only = True)
     if('median' in kind): 
         d['median']  = df.median(axis=1 , numeric_only = True) 
     if('sum' in kind): 
         d['sum']  = df.sum(axis=1 , numeric_only = True) 
     if('min' in kind): 
         d['min']  = df.min(axis=1 , numeric_only = True) 
     if('max' in kind): 
         d['max']  = df.max(axis=1 , numeric_only = True) 
     if(len(d) == 0):
         raise ValueError(f'Unrecognized kind - {kind}')

     return pd.DataFrame(d)


import pandas as pd
import numpy as np
import re

def aggregate_band(df , bandname):
    # Extract NDVI columns
    req_cols = [col for col in df.columns if col.startswith(f'{bandname}__')]
    
    # Sort NDVI columns numerically by extracting the timestep
    req_cols_sorted = sorted(req_cols, key=lambda x: int(re.search(fr'{bandname}__(\d+)', x).group(1)))
    
    # Get timestep numbers
    timesteps = np.array([int(re.search(fr'{bandname}__(\d+)', col).group(1)) for col in req_cols_sorted])
    
    # Extract the NDVI data in sorted order
    band_data = df[req_cols_sorted].values  # shape: (n_samples, n_timesteps)
    
    # Compute differences for gradient
    gradients = np.diff(band_data, axis=1)
    
    # Aggregated metrics
    mean_vals = np.nanmean(band_data, axis=1)
    median_vals = np.nanmedian(band_data, axis=1)
    max_vals = np.nanmax(band_data, axis=1)
    min_vals = np.nanmin(band_data, axis=1)
    auc_vals = np.trapz(band_data, x=timesteps, axis=1)
    max_grad_vals = np.nanmax(np.abs(gradients), axis=1)
    max_grad_indices = np.argmax(np.abs(gradients), axis=1)
    max_grad_timesteps = timesteps[1:][max_grad_indices]  # Timestep where max gradient occurred

    # Create output DataFrame
    result_df = pd.DataFrame({
        f'Mean__{bandname}': mean_vals,
        f'Median__{bandname}': median_vals,
        f'Max__{bandname}': max_vals,
        f'Min__{bandname}': min_vals,
        f'AUC__{bandname}': auc_vals,
        f'Max_Gradient__{bandname}': max_grad_vals,
        f'T_Max_Gradient__{bandname}': max_grad_timesteps
    })

    return result_df


def get_agvar(files_path , gdf ,agvar, year , agg = "mean"): 

    tiff_files = sorted([f for f in os.listdir(files_path) if f.endswith('.tif')])
    # filter by year. This is different for different agvar
    warnings.warn('A_K_ this behaviour will change with correct dates. ')
    if(agvar=='Precipitation' ): 
       tiff_files = [file for file in tiff_files if  f'v2.0.{year}' in file]
       tiff_files_new = []
       for tiff in tiff_files:
           doy = int(tiff[16:19]) 
           tiffyear = int(tiff[12:16])
           if((tiffyear == year-1 and  doy>=213) or (tiffyear == year and doy <=244)):
               tiff_files_new.append(tiff)

    elif(agvar == 'Temperature'):
        tiff_files = [file for file in tiff_files if  f'_{year}'   in file]
        tiff_files_new = []
        for tiff in tiff_files:
            doy = int(tiff[11:14]) 
            tiffyear = int(tiff[7:11])
            if((tiffyear == year-1 and  doy>=213) or (tiffyear == year and doy <=244)):
                tiff_files_new.append(tiff)
    else: 
        1/0
    assert len(tiff_files_new) >0 , "Date filter not working properly."
    tiff_files = tiff_files_new
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
