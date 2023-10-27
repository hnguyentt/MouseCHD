"""
Utilities for the package 
"""

import logging
import os
from datetime import datetime
from importlib import resources
import yaml
from functools import partialmethod
from urllib import request

import pandas as pd
import numpy as np

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".MouseCHD")
os.makedirs(CACHE_DIR, exist_ok=True)
BASE_URI = "https://imjoy-s3.pasteur.fr/public/mousechd-napari"


def set_logger(log_path):
    from imp import reload
    reload(logging)
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logging.basicConfig(
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler()],
    format='%(message)s',
    level=logging.INFO)
    
    # Start every log with the date and time
    logging.info("="*15 + "//" + "="*15)
    logging.info(datetime.now())

class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def partialclass(cls, **kwargs):
    """
    Partial class.
    """
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, **kwargs)
        
    return PartialClass


#################
# LOADING UTILS #
#################        
def load_csv_data(data_filename, data_module):
    """Load csv data from module
    
    Args:
        + data_filename (str): data filename
        + data_module (str): name of module containing data file
    Returns:
        (pd.DataFrame): pandas dataframe
    """
    with resources.open_text(data_module, data_filename) as f:
        return pd.read_csv(f)
    
    
def load_yaml_data(data_filename, data_module):
    """
    Load yaml data from module
    
    Args:
        + data_filename (str): data filename
        + data_module (str): name of module containing data file
    Returns:
        (dict): data
    """
    with resources.open_text(data_module, data_filename) as f:
        return yaml.safe_load(f)
    

def load_npy_data(data_filename, data_module):
    """
    Load npy data from module
    
    Args:
        + data_filename (str): data filename
        + data_module (str): name of module containing data file
    Returns:
        (numpy array): data
    """
    with resources.open_text(data_module, data_filename) as f:
        return np.load(f)
    
    
def get_im2arr_reader(im_ext):
    """Get image-to-array reader

    Args:
        im_ext (str): image extension
    
    Return:
        (list): A list of chained functions for image loader
    """
    import SimpleITK as sitk
    import tifffile
    import numpy as np
    
    if 'nii' in im_ext:
        return [sitk.ReadImage, sitk.GetArrayFromImage]
    elif 'tif' in im_ext:
        return [tifffile.imread]
    elif 'npy' in im_ext:
        return [np.load]
    else:
        raise NotImplementedError('"im_ex" must contain one of those strings ["nii", "tif", "npy"]')
    
    
def txt2list(path):
    """
    Read txt file and parse to list
    :param path: <str> Path to txt file
    
    :return: 
        (ls): content of the file in list format (empty if file does not exist)
    """
    if os.path.isfile(path):
        with open(path, 'r') as f:
            lines = f.read().splitlines()
    else:
        lines = []

    return lines


def list2txt(ls, path):
    """Write list to txt file

    Args:
        ls (_type_): _description_
        path (_type_): _description_
    """
    with open(path, 'a') as f:
        [f.write("%s\n" % item) for item in ls]
        

############
# DOWNLOAD # 
############
def path_to_string(path):
    # https://github.com/keras-team/keras/blob/ed99e34f279a2d2d6a44af87ee64f8fc98c7e8b9/keras/utils/io_utils.py#L85
    """Convert `PathLike` objects to their string representation.
    If given a non-string typed path object, converts it to its string
    representation.
    If the object passed to `path` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of file objects
    through this function.
    Args:
      path: `PathLike` object that represents a path
    Returns:
      A string representation of the path argument, if Python support exists.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def download_file(url,
                  fname,
                  cache_dir=CACHE_DIR,
                  update=False
                  ):
    """Download file if not exist

    Args:
        url (_type_): _description_
        fname (_type_): _description_
        cache_dir (_type_, optional): _description_. Defaults to CACHE_DIR.
    """
    outfile = os.path.join(cache_dir, fname)
    
    if update & os.path.isfile(outfile):
        os.remove(outfile)
    
    if not os.path.isfile(outfile):
        logging.info(f"Downloading {fname}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        request.urlretrieve(url, outfile)
        logging.info("Done")