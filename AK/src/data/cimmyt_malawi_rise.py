#import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
import numpy as np
import logging
import pandas as pd
import os
import rasterio as rio
import json
import sys
import geopandas as gpd

from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.data.data_loader_rise import DataLoader
from src.data.preprocess_rise import Preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parents[2]
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR')

# region to run
PATH_ID = 'malawi'

# main data paths
DATA_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "data.npy"
ID_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "ids.npy"
LABELS_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "labels.npy"
META_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "harmonized_malawi_3857_w_Id" / "harmonized_malawi_3857" / "harmonized_malawi_3857.shp"

# Time settings
TOTAL_TS = np.arange(0, 27, 1)
TS_SLICE_WIDTH = 27
TS_STEP = 1
TS_TO_PRED = []
TS_SLICES_TO_PRED = []

# Processing settings
OUTLIER_PER_TIMESTAMP = False
FIELD_MEDIAN = True
NORMALISE_STRATEGY = 'IQR'  # 'IQR','normalise'

# GT year info
YEARS = [2022, 2023]
YEAR_COLNAME = 'Year'

# Crops to drop
CROPS_TO_DROP = 'Pigeon_Pea'

# Uid column
UID = 'system:index_x'
UID_COLNAME = 'uid'

# Satellite band name and mapping to S1/S2 naming wherever possible
BAND_NAME_MAP = {
    'coastal_blue_median': 'COASTALBLUE',
    'blue_median': 'B2',
    'green_i_median': 'GREENI',
    'green_median': 'B3',
    'yellow_median': 'YELLOW',
    'red_median': 'B4',
    'rededge_median': 'REDEDGE',
    'nir_median': 'B8',
    'alpha_median': 'ALPHA'
}
BAND_NAMES = list(BAND_NAME_MAP.keys())

# Label information
GEOM_COLNAME = 'geometry'
OUT_BANDS = 2  # currently writing first 2 predicted crops, their probabilities, and a confidence measure

# Satellite indices to be used in the model (Values in dict refer to band index in the COMPLETE data to be used for making the index)
VIs = {'NDVI': [7, 5]}
# Satellite bands to be used in the model
BANDS_SELECTED = ['B2', 'GREENI', 'B3', 'YELLOW', 'B4', 'REDEDGE', 'B8']
FEATURES = BANDS_SELECTED + list(VIs.keys())

# Get the band indexes to access in the array after addition of VIs
band_names_values = list(BAND_NAME_MAP.values())
band_indices = [band_names_values.index(band) for band in BANDS_SELECTED]
vi_start_idx = len(band_names_values)
vi_indices = list(range(vi_start_idx, vi_start_idx + len(VIs)))
FEATURES_IDX = band_indices + vi_indices

class CIMMYTMalawi:
    """
    Class to handle training and inference pipeline for CIMMYT Malawi crop classification using time series raster data.
    """

    def __init__(self, RESULT_DIR: str, MODEL_DIR: str, MODE: str, mask_value: float):
        """
        Initialize class variables.

        Args:
            RESULT_DIR (str): Directory for saving results.
            MODEL_DIR (str): Path to trained model.
            MODE (str): Mode for preprocessing ('train' or 'inference').
            mask_value (float): Value to mask missing/no-data pixels.
        """
        self.RESULT_DIR = RESULT_DIR
        self.MODEL_DIR = MODEL_DIR
        self.MODE = MODE
        self.mask_value = mask_value

    def harmonise_CIMMYTMalawi_planet(self, data: np.ndarray, BAND_NAMES: list[str], BAND_NAME_MAP: dict[str, str], UID: str) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """
        Harmonises input features and creates vegetation indices.

        Args:
            data (np.ndarray): Raw data array.
            FEATURES (list[str]): List of feature names.
            BAND_NAME_MAP (dict): Mapping from raw to standardised band names.
            UID (str): Unique ID field.

        Returns:
            Tuple[np.ndarray, list[str], dict[str, str]]: Processed data, updated features list, and band name map.
        """
        logger.info('Harmonising features and adding vegetation indices')
        dataloader = DataLoader(list(BAND_NAME_MAP.keys()), self.mask_value)
        data, features, band_name_map = dataloader.make_VIs(data, VIs, BAND_NAMES, BAND_NAME_MAP)
        return data, features, band_name_map

    def apply_field_median(self, data: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the median of samples grouped by unique field IDs.

        Args:
            data (np.ndarray): Input time-series data of shape (samples, time, bands).
            ids (np.ndarray): Array of field IDs of shape (samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Median data of shape (unique_fields, time, bands), unique field IDs.
        """
        unique_ids = np.unique(ids)
        median_data = np.zeros((len(unique_ids), data.shape[1], data.shape[2]), dtype=data.dtype)

        for i, uid in enumerate(unique_ids):
            sample_mask = np.where(ids == uid)[0]
            group_samples = data[sample_mask]
            median_data[i] = np.nanmedian(group_samples, axis=0)

        return median_data, unique_ids

    def load_CIMMYTMalawi_planet(self):
        """
        Loads and processes the CIMMYT Malawi training data.

        Returns:
            tuple: (X, y, df, outlier_dict, normalise_dict, features, target, num_classes,
                    band_name_map, out_bands, features, label_encoder, total_ts)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = np.load(DATA_FILE_PATH)
        ids = np.load(ID_FILE_PATH)
        labels = np.load(LABELS_FILE_PATH)
        meta = gpd.read_file(META_FILE_PATH)

        logger.info(f"Loaded train data {data.shape}")
        logger.info(f"Loaded ids data {ids[0]}")
        logger.info(f"Loaded labels data {labels[0]}")
        logger.info(f"Value counts {meta['Intercrop'].value_counts()}")
        logger.info("Loaded polygon data")

        meta_selected = meta[meta['Source'] != 'CIMMYT'].reset_index(drop=True)
        logger.info("Filtered data according to attribute")
        logger.info(f"meta_selected head: {meta_selected.head()}")
        logger.info(f"ids shape: {ids.shape}")
        logger.info(f"data shape: {data.shape}")

        data = data[:, :, :]
        logger.info("TS selection done")

        labels = np.array(labels)
        mask = ~np.isin(labels, CROPS_TO_DROP)
        data = data[mask]
        labels = labels[mask]
        ids = ids[mask]
        logger.info(f"Dropped crops {CROPS_TO_DROP}")

        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        logger.info("Label encoding done")

        data[data == 9999.0] = self.mask_value
        logger.info("Masked no data values")

        data, _, _ = self.harmonise_CIMMYTMalawi_planet(data, BAND_NAMES, BAND_NAME_MAP, UID)
        logger.info("Added vegetation indices")
        
        preprocess = Preprocess(data, {}, {}, self.MODE, self.MODEL_DIR)
        data, outlier_dict, normalise_dict = preprocess.prep(OUTLIER_PER_TIMESTAMP, self.mask_value)
        logger.info("Preprocessing complete")

        if FIELD_MEDIAN:
            unique_ids = np.unique(ids)
            labels_encoded_median = np.array([np.median(labels_encoded[ids == uid]) for uid in unique_ids])
            data_processed, ids_processed = self.apply_field_median(data, ids)
            data = data_processed
            ids = ids_processed
            labels_encoded = labels_encoded_median
            logger.info(f"Field level data processed {data.shape, labels_encoded.shape}")
        else:
            labels_encoded = labels_encoded  # Use original encoded labels

        data = data[:, :, FEATURES_IDX]
        features = FEATURES
        logger.info("Band and VI selection done")
        logger.info(FEATURES)
        logger.info(FEATURES_IDX)

        unique, counts = np.unique(labels_encoded, return_counts=True)
        valid_labels = unique[counts >= 10]  # Ensure at least 2 samples per class
        mask_valid = np.isin(labels_encoded, valid_labels)
        data = data[mask_valid]
        labels_encoded = labels_encoded[mask_valid]
        ids = ids[mask_valid]
        logger.info(f"Filtered classes to those with >= 2 samples")

        return data, labels_encoded, le, outlier_dict, normalise_dict, features, FEATURES_IDX, TOTAL_TS, OUT_BANDS