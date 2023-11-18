import os, re
import pandas as pd
import SimpleITK as sitk 
import logging

resampled_headers = ["heart_name", 
                     "resampled_size",
                     "resampled_spacing",
                     "pixel_max",
                     "pixel_min",
                     "pixel_mean",
                     "pixel_std"]

from .utils import (load_nifti,
                    get_largest_connectivity,
                    crop_heart_bbx,
                    maskout_non_heart,
                    norm_min_max,
                    replicate,
                    split_slices,
                    pad_image)


def create_resampling_df(outdir):
    try:
        df = pd.read_csv(os.path.join(outdir, "resampled.csv"))
    except FileNotFoundError:
        df = pd.DataFrame(columns=resampled_headers)
        
    return df


def resample_folder(imdir, 
                    maskdir, 
                    outdir, 
                    metafile=None, 
                    meta_sep=",",
                    save_images=False):
    if save_images:
        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "images_x5"), exist_ok=True)
    
    df = create_resampling_df(outdir=outdir)
    filenames = [re.sub(r".nii.gz$", "", x) for x in os.listdir(maskdir) if (not x.startswith(".")) & x.endswith(".nii.gz")]
    logging.info("Number of masks in maskdir: {}".format(len(filenames)))
    
    if metafile is not None:
        meta = pd.read_csv(metafile, sep=meta_sep)
        meta = meta[meta["Stage"].isin(["E18.5", "P0", "E17.5"])]
        meta = meta[~meta["heart_name"].isin(["N_261h", "NH_229m"])] # error images
        logging.info("Number of images from metafile: {}".format(len(meta)))
        filenames = [x for x in filenames if x in meta["heart_name"].values]
        logging.info("Final number of images: {}".format(len(filenames)))
        meta.to_csv(os.path.join(outdir, os.path.basename(metafile)), index=False)
        
    df = create_resampling_df(outdir=outdir)
    filenames = [x for x in filenames if x not in df["heart_name"].tolist()]
    logging.info("Need to process: {}".format(len(filenames)))
    
    for i, heart_name in enumerate(filenames):
        logging.info("{}. {}".format(i+1, heart_name))
        
        # Load image and mask
        img = sitk.ReadImage(os.path.join(imdir, f"{heart_name}_0000"))
        im = sitk.GetArrayFromImage(img)
        spaces = img.GetSpacing()
        ma = load_nifti(os.path.join(maskdir, heart_name))
        
        # Get largest connected area
        try:
            max_clump = get_largest_connectivity(ma)
        except AssertionError:
            df.loc[len(df), :] = [heart_name] + ["Error"] * 6
            logging.info("=> Error!")
            continue
        
        cropped_im, cropped_ma = crop_heart_bbx(im, max_clump, pad=(5,5,5))
        resampled_im = maskout_non_heart(cropped_im, cropped_ma)
        resampled_im = norm_min_max(resampled_im)
        
        if save_images:
            resampled_img = sitk.GetImageFromArray(resampled_im)
            resampled_img.SetSpacing(spaces)
            
            sitk.WriteImage(resampled_img, 
                            os.path.join(outdir, "images", f"{heart_name}.nii.gz"))
            
        
        for i in range(5):
            fn = "{}_{:02d}.nii.gz".format(heart_name, i+1)
            resampled_x5 = split_slices(resampled_im, start=i, step=5, dim=0)
            resampled_img = sitk.GetImageFromArray(resampled_x5)
            resampled_img.SetSpacing((spaces[0], spaces[1], spaces[2]*5))
            sitk.WriteImage(resampled_img, os.path.join(outdir, "images_x5", fn)) 
            
        df.loc[len(df), :] = [heart_name,
                              str(resampled_im.shape),
                              str(spaces),
                              cropped_im.max(),
                              cropped_im.min(),
                              cropped_im.mean(),
                              cropped_im.std()]
        
        df.to_csv(os.path.join(outdir, "resampled.csv"), index=False)    
        
        
        
