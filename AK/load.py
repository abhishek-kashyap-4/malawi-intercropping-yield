#generic
from pathlib import Path
import numpy as np
import pandas as pd 
import logging 

#ML 

#RS 

#Project


## Inits 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parents[0]

# main data paths
DATA_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "data.npy"
ID_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "ids.npy"
LABELS_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "labels.npy"
META_FILE_PATH = base_path / "data" / "train_ready" / "malawi_triplet_eos" / "train_rise" / "harmonized_malawi_3857_w_Id" / "harmonized_malawi_3857" / "harmonized_malawi_3857.shp"




