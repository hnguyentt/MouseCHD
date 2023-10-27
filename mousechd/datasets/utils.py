"""
UTILITY FUNCTIONS 
"""
import logging
import numpy as np
import pandas as pd
import tifffile
import SimpleITK as sitk
import nrrd
from skimage.measure import label
import itertools
import re
import os
import requests

from mousechd.utils.tools import load_csv_data

DATA_MODULE = "mousechd.datasets.data"

EXCLUDE_IDS = ["174", "183", "184", "185",
               "190", "206", "207", "208",
               "N_261h", "NH_229m"]

RESAMPLE_METHODS = ["plain", "rotx90", "rotx180", "rotx270",
                    "roty90", "roty180", "roty270",
                    "rotz90", "rotz180", "rotz270",
                    "rotx270y180", "rotx90z90", "rotx90z180",
                    "rotx90z270", "rotx180y90", "rotx180y90z90", 
                    "roty90z180", "rotx180y270z270", "nocrop"]

REPLICATE = ["None", "x", "iso"]

INTEREST_FIELDS = ["folder", "heart_name", "origin_view", "size", "spacing", "heartmask"]

# <-----------------IMAGE UTILS-------------------> #
def get_series_filenames(path):
    """Get filenames of the same series in folder <path>
    Sometimes, the dicom folder doesn't contain the homogeneous series,
    It may contains a series of different slices of CT scans as well as the ScootView,
    which looks similar to Xray.
    Args:
        path (str): dicom folder path
    Returns:
        (tuple): a tuple of filenames of the same series
    """
    reader = sitk.ImageSeriesReader()
    series_file_names = {}
    series_IDs = reader.GetGDCMSeriesIDs(path)
    
    for series in series_IDs:
        series_file_names[series] = reader.GetGDCMSeriesFileNames(path, series)
        
    series = []
    for k, v in series_file_names.items():
        series.append((len(v),k))
    
    series = sorted(series, reverse=True)
    
    return series_file_names[series[0][1]]

def dicom2nii(path):
    """Read dicom folder path and return ITK image
    Args:
        path (str): path to dcom folder
    Return:
        img: ITK image
    """
    reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(path) # only choose the first series, will fail when there are more than 1 series in path
    dicom_names = get_series_filenames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    return image


def nrrd2nii(path, orientation="SAR", spacing=(0.02, 0.02, 0.02)):
    
    im, header = nrrd.read(path)
    img = sitk.GetImageFromArray(im)
    del im
    img.SetSpacing(header.get('spacing', spacing))
    img.SetDirection(sitk.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation(orientation))
    
    return img


# <----------------SEGMENTATION UTILS-------------------> #
def get_largest_connectivity(segmentation):
    """Find largest connected component from segmentation

    Args:
        segmentation (numpy array): Segmentation
    Returns:
        numpy array: largest connected component
    """
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return (largestCC*1).astype(np.uint8)

def get_translate_values(seg_arr, pad=(0,0,0)):
    """Get translation values for image and segmentation
    Args:
        im_arr (np array): numpy array of image
        seg_arr (np array): numpy array of segmentation
    Returns:
        np array: translation values
    """
    heart_pos = np.where(seg_arr != 0)
    
    min_z = max(0,heart_pos[0].min()-pad[0])
    min_y = max(0,heart_pos[1].min()-pad[1])
    min_x = max(0,heart_pos[2].min()-pad[2])
    
    return min_z, min_y, min_x

def get_bbx(seg_arr, pad=(0,0,0)):
    """
    Get bounding box arround the heart
    
    Args:
        seg_arr (np.array): 3D-array of heart mask
        pad (3-element list-like): padding values
    
    Returns:
        tuple of bounding box around heart (min_z, max_z), (min_y, max_y), (min_x, max_x) 
    """
    heart_pos = np.where(seg_arr != 0)
    
    min_z, max_z = max(0,heart_pos[0].min()-pad[0]), min(heart_pos[0].max()+pad[0], seg_arr.shape[0])
    min_y, max_y = max(0,heart_pos[1].min()-pad[1]), min(heart_pos[1].max()+pad[1], seg_arr.shape[1])
    min_x, max_x = max(0,heart_pos[2].min()-pad[2]), min(heart_pos[2].max()+pad[2], seg_arr.shape[2])
    
    return (min_z, max_z), (min_y, max_y), (min_x, max_x)
    

def crop_heart_bbx(im_arr, seg_arr, pad=(0,0,0)):
    """Cropping ct scan with bounding box defined by segmentation
    Args:
        im_arr (np array): numpy array of image
        seg_arr (np array): numpy array of segmentation
        pad (int, optional): padding. Defaults to 2.
    Returns:
        np arrays: cropped_im_arr, cropped_seg_arr
    """
    heart_pos = np.where(seg_arr != 0)
    
    min_z, max_z = max(0,heart_pos[0].min()-pad[0]), min(heart_pos[0].max()+pad[0], im_arr.shape[0])
    min_y, max_y = max(0,heart_pos[1].min()-pad[1]), min(heart_pos[1].max()+pad[1], im_arr.shape[1])
    min_x, max_x = max(0,heart_pos[2].min()-pad[2]), min(heart_pos[2].max()+pad[2],im_arr.shape[2])
    
    cropped_seg_arr = seg_arr[min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_im_arr = im_arr[min_z:max_z, min_y:max_y, min_x:max_x]
    
    return cropped_im_arr.copy(), cropped_seg_arr.copy()

def load_slices(path):
    slice_ls = sorted([x for x in os.listdir(path)], reverse=True)
    scan = []
    for x in slice_ls:
        im = tifffile.imread(os.path.join(path, x))
        scan.append(im)
        # scan.append(np.rot90(np.rot90(im)))
    
    return np.asarray(scan)

def load_tif3d(path):
    """Load 3D segmentation from path
    Args:
        path (str): path to segmentation
    Returns:
        np array: segmentation
    """
    mask_arr = tifffile.imread(f"{path}.tif")
    mask_arr = mask_arr[::-1]
    
    return mask_arr

def load_nifti(path):
    """Load segmentation from path
    Args:
        path (str): path to segmentation
    Returns:
        np array: segmentation
    """
    
    return sitk.GetArrayFromImage(sitk.ReadImage(f'{path}.nii.gz'))


def pad_image(im):
    """
    Padding image to make it cube.
    Args:
        im (np.array): image array
        
    Returns:
        np.array: padded image array
    """
    max_dim = max(im.shape)
    pads = [((max_dim - im.shape[i])//2,
             max_dim - im.shape[i] - (max_dim - im.shape[i])//2)
            for i in range(len(im.shape))]
    
    return np.pad(im, pads, mode='constant')


# <-----------------STANDARDIZING UTILS-------------------> #
def mirror(img, seg):
    """
    Mirror Direction, Origin, Spacing from img to seg
    :params
        img: <sitk image> SimpleITK image
        seg: <sitk image | numpy array> SimpleITK image or numpy array of segmentation
    :return:
        seg: <sitk image> with Direction, Origin, Spacing the same as img
    """   
    if type(seg) == np.ndarray:
        seg = sitk.GetImageFromArray(seg)
        
    seg.SetOrigin(img.GetOrigin())
    seg.SetDirection(img.GetDirection())
    seg.SetSpacing(img.GetSpacing())
    
    return seg

def anyview2LPS(img):
    """Convert any views: PIR, LPI, RPI, etc. to LPS
    Args:
        img (SimpleITK image): SimpleITK image to be converted to LPS
    Return:
        new_img: image in LPS view
    """
    direction = np.asarray(img.GetDirection())
    orig_spacing = np.asarray(img.GetSpacing())
    
    if len(direction) != 9:
        return "Undefined"
    else:
        if sum(direction[[2,3,7]] != 0) == 3:
            im = sitk.GetArrayFromImage(img)
            im = np.transpose(im,(1,2,0))
            im = np.flip(im,np.where(direction[[2,3,7]][::-1]<0)[0])
            
            new_img = sitk.GetImageFromArray(im)
            new_img.SetOrigin(img.GetOrigin())
            new_img.SetDirection((1,0,0,0,1,0,0,0,1))
            new_img.SetSpacing([orig_spacing[2],orig_spacing[0],orig_spacing[1]])
            
            return new_img
        
        elif sum(direction[[0,5,7]] != 0) == 3:
            im = sitk.GetArrayFromImage(img)
            im = np.transpose(im,(1,0,2))
            im = np.flip(im,np.where(direction[[0,5,7]][::-1]<0)[0])
            
            new_img = sitk.GetImageFromArray(im)
            new_img.SetOrigin(img.GetOrigin())
            new_img.SetDirection((1,0,0,0,1,0,0,0,1))
            new_img.SetSpacing([orig_spacing[0],orig_spacing[2],orig_spacing[1]])
            
            return new_img
        
        elif  sum(direction[[0,4,8]] != 0) == 3:
            im = sitk.GetArrayFromImage(img)
            im = np.flip(im,np.where(direction[[0,4,8]][::-1]<0)[0])
            
            new_img = sitk.GetImageFromArray(im)
            new_img.SetOrigin(img.GetOrigin())
            new_img.SetDirection((1,0,0,0,1,0,0,0,1))
            new_img.SetSpacing(orig_spacing)
            
            return new_img
        
        elif sum(direction[[2,4,6]] != 0) == 3:
            im = sitk.GetArrayFromImage(img)
            im = np.transpose(im, (2,1,0))
            im = np.flip(im,np.where(direction[[2,4,6]][::-1]<0)[0])
            
            new_img = sitk.GetImageFromArray(im)
            new_img.SetOrigin(img.GetOrigin())
            new_img.SetDirection((1,0,0,0,1,0,0,0,1))
            new_img.SetSpacing(orig_spacing)
            
            return new_img
        
        else:
            return "Undefined"
        
def get_view_sitk(img):
    """Get view from SITK image: LSP, LPS, etc.
    Args:
        img (sitk image): image read by SimpleITK package
    Returns:
        (str): view
    """
    view = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())

    return view


def make_isotropic(image,
                   interpolator=sitk.sitkLinear,
                   spacing=None,
                   default_value=0,
                   standardize_axes=False):
    """
    https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/05_Results_Visualization.ipynb
    Make an image isotropic via resampling if needed.
    
    Args:
        image (sitk.Image): image to make isotropic
        interpolator (sitk.sitkInterpolator): interpolator to use (default: sitk.sitkLinear)
        spacing (int): spacing to use (default: None)
        default_value (image.GetPixelID): desired pixel value for resampled points that fall outside the original image (default: 0)
        standardize_axes (bool): whether to standardize the axes (default: False)
    Returns:
        SimpleITK.Image: isotropic image which occupies the same region in space as the input image
    """
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    new_spacing = [spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()
    # Only need to standardize axes if user requested and the original
    # axes were not standard.
    if standardize_axes and not np.array_equal(
        np.array(new_direction), np.identity(image.GetDimension()).ravel()
    ):
        new_direction = np.identity(image.GetDimension()).ravel()
        # Compute bounding box for the original, non standard axes image.
        boundary_points = []
        for boundary_index in list(itertools.product(*zip([0, 0, 0], image.GetSize()))):
            boundary_points.append(image.TransformIndexToPhysicalPoint(boundary_index))
        max_coords = np.max(boundary_points, axis=0)
        min_coords = np.min(boundary_points, axis=0)
        new_origin = min_coords
        new_size = (((max_coords - min_coords) / spacing).round().astype(int)).tolist()
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )


# <-----------------RESAMPLE UTILS-------------------> #
def maskout_non_heart(img, seg):
    im = img.copy()
    im[seg == 0] = 0
    im[seg == 0] = im[seg != 0].min() - 1
    
    return im


def norm_min_max(im):
    im_arr = im.copy()
    
    im_arr = (im_arr - im_arr.min()) / (im_arr.max() - im_arr.min())
    
    return im_arr.astype(np.float32)


def split_slices(im, start=0, step=5, dim=0):
    """Split one hearts into multiple parts

    Args:
        im (np.array): 3d image
        start (int, optional): start index. Defaults to 0.
        step (int, optional): interval. Defaults to 5.
        dim (int, optional): dimension to slice. Defaults to 0.

    Returns:
        np.array: image part
    """
    assert dim < 3, "Dimension must be < 3"
    ids = np.arange(start=start, stop=im.shape[dim], step=step)
    
    if dim==0:
        return im[ids]
    elif dim==1:
        return im[:,ids,:]
    else:
        return im[:,:,ids]
    
def replicate(im, start=0, step=5, method="x"):
    """Replicate 3D image to many 3D images by skipping slices.
    Args:
        im (np.array): image array
        start (int, optional): start index. Defaults to 0.
        step (int, optional): skipping step. Defaults to 5.
        method (str, optional): replicate method: 'x' means only replicate along x axis, 'iso' means replicate along 3 axes. Defaults to "x".

    Returns:
        np.array: index array
    """
    assert method in ["x", "iso"], "Method must be 'x' or 'iso'"
    if method == "x":
        return split_slices(im, start=start, step=step, dim=0)
    else:
        im_rep = split_slices(im, start=start, step=step, dim=0)
        im_rep = split_slices(im_rep, start=start, step=step, dim=1)
        im_rep = split_slices(im_rep, start=start, step=step, dim=2)
        
        return im_rep
    
    
def replicate_itk(impath, rep_factor, df=None, method="x", **kwargs):
    save = kwargs.get('save', None)
    save_df = kwargs.get('save_df', None)
    sep = kwargs.get('sep', ',')
    
    img = sitk.ReadImage(impath)
    im = sitk.GetArrayFromImage(img)
    heart_name = re.sub(r'.nii.gz$', '', os.path.basename(impath))
    
    sizes = []
    spaces = []
    pixel_max = []
    pixel_min = []
    pixel_mean = []
    pixel_std = []
         
    for i in range(rep_factor):
        filename = "{}_{:02d}.nii.gz".format(heart_name, i+1)
        resampled_im = replicate(im=im, start=i, step=rep_factor, method=method)
        resampled_img = sitk.GetImageFromArray(resampled_im)
        
        spacing = img.GetSpacing()
        if method == "x":
            resampled_img.SetSpacing((spacing[0], spacing[1], spacing[2]*rep_factor))
        else:
            resampled_img.SetSpacing((spacing[0]*rep_factor, spacing[1]*rep_factor, spacing[2]*rep_factor))
        
        if save is not None:
            sitk.WriteImage(resampled_img, os.path.join(save, filename))
        
        sizes.append(resampled_img.GetSize())
        spaces.append(resampled_img.GetSpacing())
        pixel_max.append(np.max(resampled_im))
        pixel_min.append(np.min(resampled_im))
        pixel_mean.append(np.mean(resampled_im))
        pixel_std.append(np.std(resampled_im))
        
    if df is not None:
        df.loc[heart_name, "resampled_size"] = str(sizes)
        df.loc[heart_name, "resampled_spacing"] = str(spaces)
        df.loc[heart_name, "pixel_max"] = str(pixel_max)
        df.loc[heart_name, "pixel_min"] = str(pixel_min)
        df.loc[heart_name, "pixel_mean"] = str(pixel_mean)
        df.loc[heart_name, "pixel_std"] = str(pixel_std)
        
        if save_df is not None:
            df.to_csv(save_df, sep=sep)
    
    return resampled_img
    
    
def rotate3d(image, theta_x=0, theta_y=0, theta_z=0, output_spacing = None, background_value=0.0):
    """
    https://discourse.itk.org/t/simpleitk-euler3d-transform-problem-with-output-size-resampling/4387/4?u=nguyenhoa93
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively and resamples it to be isotropic.
    :param image: An sitk 3D image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param output_spacing: Scalar denoting the isotropic output image spacing. If None, then use the smallest
                           spacing from original image.
    :return: The rotated image
    
    Examples
    --------
    Let's say we have a SimpleITK 3D image and want to rotate it by 30 degrees around the x axis
    >>> from mousechd.dataset.utils import rotate3d
    >>> rot_img = rotate3d(image, theta_x=30)
    """
    euler_transform = sitk.Euler3DTransform (image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2.0 for sz in image.GetSize()]), 
                                             np.deg2rad(theta_x), 
                                             np.deg2rad(theta_y), 
                                             np.deg2rad(theta_z))

    # compute the resampling grid for the transformed image
    max_indexes = [sz-1 for sz in image.GetSize()]
    extreme_indexes = list(itertools.product(*(list(zip([0]*image.GetDimension(),max_indexes)))))
    extreme_points_transformed = [euler_transform.TransformPoint(image.TransformContinuousIndexToPhysicalPoint(p)) for p in extreme_indexes]
    
    output_min_coordinates = np.min(extreme_points_transformed, axis=0)
    output_max_coordinates = np.max(extreme_points_transformed, axis=0)
    
    # isotropic ouput spacing
    if output_spacing is None:
        output_spacing = min(image.GetSpacing())
    output_spacing = [output_spacing]*image.GetDimension()  

    output_origin = output_min_coordinates
    output_size = [int(((omx-omn)/ospc)+0.5)  for ospc, omn, omx in zip(output_spacing, output_min_coordinates, output_max_coordinates)]
    
    # direction cosine of the resulting volume is the identity (default)
    return sitk.Resample(image1 = image, 
                         size = output_size, 
                         transform = euler_transform.GetInverse(), 
                         interpolator = sitk.sitkLinear, 
                         outputOrigin = output_origin,
                         outputSpacing = output_spacing,
                         defaultPixelValue = background_value)
    
    
def resample3d(im, size=(224,224,64), interpolator = sitk.sitkNearestNeighbor):
    ori_spacing = im.GetSpacing()
    ori_size = im.GetSize()
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(im.GetDirection())
    resampler.SetSize(size)
    resampler.SetOutputSpacing([ori_size[i]*ori_spacing[i]/size[i] for i in range(3)])
    
    resampled_im = resampler.Execute(im)
    
    return resampled_im

    
def crop2dimage(im, left_off=0, right_off=0, top_off=0, bottom_off=0):
    """
    Crop 2d-image (color or gray scale) with left, right, top, and bottom offset
    
    Args:
        im (np.array): 2-D (H x W) or 3-D (H x W x C) numpy array
        left_off (float): left offset
        right_off (float): right offset
        
    """
    assert len(im.shape) in [2, 3], "Expected 2-dim or 3-dim array, got {}".format(len(im.shape))
    assert (left_off + right_off) < 1, 'left_off + right_off must be < 1'
    assert (top_off + bottom_off) < 1, 'top_off + bottom_off must be < 1'
    left_off = int(im.shape[1]*left_off)
    right_off = max(int(im.shape[1]*right_off), 1)
    top_off = int(im.shape[0]*top_off)
    bottom_off = max(int(im.shape[0]*bottom_off), 1)
    
    if len(im.shape)==3:
        return im[top_off:-bottom_off, left_off:-right_off,:]
    else:
        return im[top_off:-bottom_off, left_off:-right_off]
        
# <-----------------MISC UTILS-------------------> #
def get_latest_meta(path, prefix):
    # Get latest metadata file
    all_meta = [x for x in os.listdir(path) if bool(re.match(r"^{}_\d+.csv$".format(prefix), x))]
    dates = [int(x.split(".")[0].split("_")[-1]) for x in all_meta]
    latest_idx = dates.index(max(dates))
    
    return os.path.join(path, all_meta[latest_idx])

def save_fold_labels(save_dir, kfolds, prefix=None):
    """Saving fold labels to csv file

    Args:
        save_dir (str): path to save directory
        kfolds (list): list of fold labels, each item in the list should be a dictionary containing keys: "train", "val", "test"; the values are dataframe of fold labels
    Returns: None
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, fold in enumerate(kfolds):
        os.makedirs(os.path.join(save_dir, f'F{i+1}'), exist_ok=True)
        if prefix is None:
            fold["train"].to_csv(os.path.join(save_dir, f"F{i+1}", "train.csv"), index=False)
            fold["val"].to_csv(os.path.join(save_dir, f"F{i+1}", "val.csv"), index=False)
            fold["test"].to_csv(os.path.join(save_dir, f"F{i+1}", "test.csv"), index=False)
        else:
            fold["train"].to_csv(os.path.join(save_dir, f"F{i+1}", f"{prefix}_train.csv"), index=False)
            fold["val"].to_csv(os.path.join(save_dir, f"F{i+1}", f"{prefix}_val.csv"), index=False)
            fold["test"].to_csv(os.path.join(save_dir, f"F{i+1}", f"{prefix}_test.csv"), index=False)
        
def get_origin_heartname(x):
    x = x.split(os.sep)[-1]
    
    return re.sub(r"_0[1-5]$", "", x)


def get_stats_df(df, meta_df):
    logging.info(df["label"].value_counts())
    df["heart"] = df["heart_name"].apply(get_origin_heartname)
    logging.info("# hearts: {}".format(len(df["heart"].unique())))
    df = df.drop_duplicates(["heart"], keep="first")
    df["stage"] = df["heart"].map(meta_df.set_index("heart_name").to_dict()["Stage"])
    logging.info(pd.crosstab(df["stage"], df["label"]))
    

def get_stats_kfolds(dfs, meta_df):
    for i, fold in enumerate(dfs):
        logging.info("-"*10)
        logging.info("Fold {}".format(i+1))
        logging.info("===>TRAIN")
        get_stats_df(df=fold["train"].copy(), meta_df=meta_df)
        # ---
        logging.info("===>VAL")
        get_stats_df(df=fold["val"].copy(), meta_df=meta_df)
        # ---
        logging.info("===>TEST")
        get_stats_df(df=fold["test"].copy(), meta_df=meta_df)
        

def add_prefix_kfolds(dfs, prefix):
    kfolds = []
    for fold in dfs:
        train_df = fold["train"]
        train_df["heart_name"] = prefix + train_df["heart_name"]
        test_df = fold["test"]
        test_df["heart_name"] = prefix + test_df["heart_name"]
        val_df = fold["val"]
        val_df["heart_name"] = prefix + val_df["heart_name"]
        
        kfolds.append({"train": train_df, "test": test_df, "val": val_df})
        
    return kfolds
    


