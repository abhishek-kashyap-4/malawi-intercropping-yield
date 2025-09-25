from typing import Optional, Dict, Tuple, Union, Any
from scipy.signal import savgol_filter
import numpy as np
import json
import numba
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@numba.njit(parallel=True)
def _mask_interpolate_2D(band_data_2D: np.ndarray) -> np.ndarray:
    """
    Interpolates missing values (np.nan) in a 2D array along axis=1 (time).
    
    Args:
        band_data_2D (np.ndarray): 2D array with shape (samples, time).
        
    Returns:
        np.ndarray: Interpolated 2D array of the same shape as input.
    """
    n_samples, n_timestamps = band_data_2D.shape
    band_interp_data_2D = band_data_2D.copy()

    for n_sample in numba.prange(n_samples):
        valid_indexes = np.where(~np.isnan(band_data_2D[n_sample]))[0]
        if valid_indexes.size == 0:
            band_interp_data_2D[n_sample] = 0  
            continue
        if valid_indexes.size == n_timestamps:
            continue
        fp = band_data_2D[n_sample, valid_indexes]
        band_interp_data_2D[n_sample] = np.interp(
            np.arange(n_timestamps),
            valid_indexes,
            fp
        )
    return band_interp_data_2D


def _flatten_2D(arr: np.ndarray) -> np.ndarray:
    """
    Flattens an N-dimensional array into 2D by collapsing all but last dimension.

    Args:
        arr (np.ndarray): Input array of shape (n1, n2, ..., nk, n_t).

    Returns:
        np.ndarray: 2D array of shape (n1*n2*...*nk, n_t).
    """
    *n_rem, n_ts = arr.shape
    return arr.reshape((np.prod(n_rem), n_ts))


def mask_interpolate(
    band_data: np.ndarray,
    mask_value: float = np.nan,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Interpolates missing values (np.nan and mask_value) across time in an N-D array.

    Args:
        band_data (np.ndarray): Array with shape (samples, time) or similar.
        mask_value (float): Value considered as missing (default 65535).
        n_jobs (int): Number of threads for parallelization; -1 to use all CPUs.

    Returns:
        np.ndarray: Same shape as input with missing values interpolated.
    """
    band_data = np.array(band_data, dtype=np.float32)
    logger.info(f"Count of NaNs before replacement: {np.isnan(band_data).sum()}")
    if np.isnan(mask_value):
        pass
    else:
        band_data = band_data.astype(np.float32)
        band_data = np.where(band_data == mask_value, np.nan, band_data)
    if not np.isnan(band_data).any():
        return band_data
    band_data_2D = _flatten_2D(band_data)
    
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    numba.set_num_threads(n_jobs)
    band_data_2D_interp = _mask_interpolate_2D(band_data_2D)
    logger.info(f"Count of NaNs after interp: {np.isnan(band_data_2D_interp).sum()}")
    return band_data_2D_interp.reshape(band_data.shape)

class Preprocess:
    """
    Preprocessing class to remove outliers, interpolate missing values,
    smooth time series, and normalize features.

    Attributes:
        data (np.ndarray): Data array with shape (samples, time, bands).
        outlier_dict (Dict): Stores outlier bounds for each band (and timestamp if per_timestamp).
        normalise_dict (Dict): Stores normalization bounds for each band.
        mode (str): Either 'train' or 'inference'.
        MODEL_DIR (Optional[str]): Directory path to save/load preprocessing metadata.
    """

    def __init__(
        self,
        data: np.ndarray,
        outlier_dict: Optional[Dict] = None,
        normalise_dict: Optional[Dict] = None,
        mode: Optional[str] = None,
        MODEL_DIR: Optional[str] = None,
    ):
        self.data = data
        self.outlier_dict = outlier_dict if outlier_dict is not None else {}
        self.normalise_dict = normalise_dict if normalise_dict is not None else {}
        self.mode = mode
        self.MODEL_DIR = MODEL_DIR

    def remove_outliers(self, band_index: int, per_timestamp: bool = False, mask_value: int = 65535) -> None:
        """
        Removes outliers by setting values outside IQR bounds to NaN.
        Can operate globally across all timestamps or per timestamp.

        Args:
            band_index (int): Index of the band to process.
            per_timestamp (bool): If True, detect outliers per timestamp.
                                  If False, detect globally across all timestamps.
        """
        band_data = self.data[:, :, band_index]  # shape (samples, time)

        if self.mode == 'train':
            if not per_timestamp:
                # Global outlier removal
                q1 = np.nanpercentile(band_data, 25)
                q3 = np.nanpercentile(band_data, 75)
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr

                mask_outliers = (band_data < lb) | (band_data > ub)
                band_data[mask_outliers] = mask_value

                self.outlier_dict[band_index] = [float(round(lb, 2)), float(round(ub, 2))]

            else:
                # Per timestamp outlier removal
                n_timestamps = band_data.shape[1]
                band_outliers = {}
                for t in range(n_timestamps):
                    vals = band_data[:, t]
                    q1 = np.nanpercentile(vals, 25)
                    q3 = np.nanpercentile(vals, 75)
                    iqr = q3 - q1
                    lb = q1 - 1.5 * iqr
                    ub = q3 + 1.5 * iqr

                    mask_outliers = (vals < lb) | (vals > ub)
                    band_data[mask_outliers, t] = np.nan

                    band_outliers[t] = [float(round(lb, 2)), float(round(ub, 2))]
                self.outlier_dict[band_index] = band_outliers

            self.data[:, :, band_index] = band_data

        elif self.mode == 'inference':
            if not per_timestamp:
                lb, ub = self.outlier_dict[band_index]
                mask_outliers = (band_data < lb) | (band_data > ub)
                band_data[mask_outliers] = np.nan
                self.data[:, :, band_index] = band_data
            else:
                band_outliers = self.outlier_dict[band_index]
                for t in range(band_data.shape[1]):
                    lb, ub = band_outliers.get(t, (None, None))
                    if lb is None or ub is None:
                        continue
                    vals = band_data[:, t]
                    mask_outliers = (vals < lb) | (vals > ub)
                    band_data[mask_outliers, t] = np.nan
                self.data[:, :, band_index] = band_data

    def normalise_features(self, band_data, band_index: int) -> None:
        """
        Normalizes the feature band data to the range [-1, 1].

        Args:
            band_index (int): Index of the band to normalize.
        """

        if self.mode == 'train':
            finite_vals = band_data[np.isfinite(band_data)]
            lb = np.min(finite_vals)
            ub = np.max(finite_vals)
            denom = ub - lb if ub != lb else 1e-6

            normed = ((2 * (band_data - lb)) / denom) - 1
            self.normalise_dict[band_index] = [float(round(lb, 2)), float(round(ub, 2))]

        elif self.mode == 'inference':
            lb, ub = self.normalise_dict[band_index]
            denom = ub - lb if ub != lb else 1e-6
            normed = ((2 * (band_data - lb)) / denom) - 1
        return normed
    
    def prep(
        self,
        outlier_per_timestamp: bool = False,
        mask_value: float = np.nan
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Runs outlier removal, interpolation, smoothing, and normalization on all bands.

        Args:
            outlier_per_timestamp (bool): If True, do outlier detection per timestamp.

        Returns:
            Tuple[np.ndarray, Dict, Dict]: Processed data, outlier dict, normalization dict.
        """

        n_samples, n_timestamps, n_bands = self.data.shape
        processed_data = np.empty((n_samples, n_timestamps, n_bands), dtype=np.float32)

        for band_index in range(n_bands):
            logger.info(f"Preprocessing. Band : {band_index}")
            
            # Outlier removal
            self.remove_outliers(band_index, per_timestamp=outlier_per_timestamp, mask_value=mask_value)
            logger.info(f"Outliers clipped. Method : {outlier_per_timestamp}")

            # Interpolate missing values (NaNs)
            band_data = self.data[:, :, band_index]
            logger.info(f"Masking value {mask_value}")
            logger.info(f"pre int : min', {self.data[:, :, band_index].min()}")
            logger.info(f"pre int : max', {self.data[:, :, band_index].max()}")
            band_data = mask_interpolate(band_data, mask_value)
            self.data[:, :, band_index] = band_data
            logger.info(f"post int : min', {band_data.min()}")
            logger.info(f"post int : max', {band_data.max()}")
            logger.info(f"Missing values interpolated")
            
            # Savitzky-Golay filter smoothing along time axis
            band_data = savgol_filter(
                self.data[:, :, band_index].astype(np.float32), 5, 2, axis=1, mode='nearest'
            )
            logger.info(f"post sav : min', {band_data.min()}")
            logger.info(f"post sav : max', {band_data.max()}")
            logger.info(f"Smoothened using Sav-Gol. Params : Window 5, Polyorder 2")

            # Normalise
            band_data = self.normalise_features(
            band_data,
            band_index
            )
            logger.info(f"post norm : min', {band_data.min()}")
            logger.info(f"post norm : max', {band_data.max()}")
            processed_data[:, :, band_index] = band_data
            logger.info(f"Normalised to -1 to 1")
            print("\n\n")

        # Save dicts if training and MODEL_DIR set
        if self.mode == 'train' and self.MODEL_DIR is not None:
            with open(f"{self.MODEL_DIR}/outlier_dict.json", 'w') as f_out:
                json.dump(self.outlier_dict, f_out)
            with open(f"{self.MODEL_DIR}/normalise_dict.json", 'w') as f_out:
                json.dump(self.normalise_dict, f_out)
        
        del self.data
        self.data = None
        return processed_data, self.outlier_dict, self.normalise_dict
    
    # def sample(self, X, y, sampling_technique, sampling_mode, n=None):
    #     from imblearn.under_sampling import RandomUnderSampler
    #     from imblearn.pipeline import Pipeline
    #     from collections import Counter
        
    #     if sampling_technique == 'SMOTE':
    #         if sampling_mode == 'majority':
    #             sampling_strategy = 'auto'
    #             smote = SMOTE(random_state=42,sampling_strategy=sampling_strategy)
    #             X_resampled, y_resampled = smote.fit_resample(X, y)
    #         elif sampling_mode == 'minority':
    #             sampling_strategy = 'minority'
    #             smote = SMOTE(random_state=42,sampling_strategy=sampling_strategy)
    #             X_resampled, y_resampled = smote.fit_resample(X, y)
    #         elif sampling_mode == 'fixed':
    #             if n == None:
    #                 raise ValueError("specify a fixed number to sample")
    #             else:
    #                 sampling_strategy_under = {}
    #                 sampling_strategy_over = {}
    #                 classes_dict = dict(Counter(y))

    #                 for k,v in classes_dict.items():
    #                     if v>=n:
    #                         sampling_strategy_under[k] = n
    #                     else:
    #                         sampling_strategy_over[k] = n

    #                 smote = SMOTE(random_state=42,sampling_strategy=sampling_strategy_over)
    #                 smote = RandomOverSampler(sampling_strategy=sampling_strategy_over, random_state=42)
    #                 under = RandomUnderSampler(sampling_strategy=sampling_strategy_under) 
    #                 pipeline = Pipeline(steps=[('smote', smote), ('under', under)])
    #                 X_resampled, y_resampled = pipeline.fit_resample(X, y)
    #         else:
    #             raise ValueError("Sampling mode is not valid. try 'majority','minority' or 'fixed'")
    #     elif sampling_technique == 'Resample':
    #         unique, counts = np.unique(y, return_counts=True)
    #         class_counts = dict(zip(unique, counts))
            
    #         if sampling_mode == 'majority':
    #             n = max(class_counts.values())
    #         elif sampling_mode == 'minority':
    #             n = min(class_counts.values())
    #         elif sampling_mode == 'fixed':
    #             if n == None:
    #                 raise ValueError("specify a fixed number to sample")
    #         else:
    #             raise ValueError("Sampling mode is not valid. try 'majority','minority' or 'fixed'")

    #         classes = np.unique(y)
    #         X_resampled = []
    #         y_resampled = []
            
    #         for c in classes:
    #             X_c = X[y == c]
    #             y_c = y[y == c]
    #             X_c_resampled, y_c_resampled = resample(X_c, y_c, n_samples=n, replace=True)  # Use replace=False if you want no duplicates

    #             X_resampled.append(X_c_resampled)
    #             y_resampled.append(y_c_resampled)

    #         X_resampled = np.vstack(X_resampled)
    #         y_resampled = np.hstack(y_resampled)
    #     else:
    #         raise ValueError('Sampling Strategy does not exist')
    #     return X_resampled, y_resampled