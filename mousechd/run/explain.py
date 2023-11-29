import sys, os, json
import re
import glob
import numpy as np
import pandas as pd
import tifffile
import cv2
import SimpleITK as sitk
import tensorflow as tf
import logging 

from mousechd.utils.tools import BASE_URI, set_logger
from mousechd.classifier.gradcam import GradCAM3D, overlay3d, generate_parellel
from mousechd.classifier.datagens import MouseCHDEvalGen
from mousechd.classifier.models import load_MouseCHD_model
from mousechd.classifier.utils import download_clf_models
from mousechd.datasets.utils import norm_min_max, resample3d, get_largest_connectivity


def add_args(parser):
    parser.add_argument("-exp_dir", type=str, help="experiment directory", default=os.path.join(BASE_URI))
    parser.add_argument("-imdir", type=str, help="image directory")
    parser.add_argument("-maskdir", type=str, help="mask directory", default=None)
    parser.add_argument("-layer_name", type=str, help="layer name to generate GradCAM", default=None)
    parser.add_argument("-outdir", type=str, help="output directory", default=None)
    parser.add_argument("-ckpt", type=str, help="checkpoint to restore", default="best_model.hdf5")
    parser.add_argument("-label_path", type=str, help="path to label", default=None)
    

def main(args):
    # Create output directories
    if args.outdir is None:
        outdir = os.path.join(args.exp_dir, "GradCAM", args.ckpt)
    else:
        outdir = args.outdir
    os.makedirs(os.path.join(outdir, "positives"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "negatives"), exist_ok=True)
    
    set_logger(os.path.join(outdir, "explain.log"))
    
    # Load model
    download_clf_models()
    model = load_MouseCHD_model(conf_path=os.path.join(args.exp_dir, "configs.json"),
                                weights_path=os.path.join(args.exp_dir, args.ckpt))
    input_shape = model.layers[0].output_shape[0][1:4]
    
    grad_model = GradCAM3D(model, layerName=args.layer_name)
    
    if args.label_path is not None:
        df = pd.read_csv(args.label_path)
        imls = df["heart_name"].to_list()
    else:
        imls = [re.sub(r".nii.gz$", "", x) for x in os.listdir(args.imdir) if x.endswith(".nii.gz") and (not x.startswith("."))]
    
    logging.info("Total: {}".format(len(imls)))
    processed = glob.glob(os.path.join(outdir, "*", "*", "parallel_sagittal.tif"))
    processed = [x.split(os.sep)[-2] for x in processed]
    logging.info("Processed: {}".format(len(processed)))
    imls = [x for x in imls if x not in processed]
    logging.info("Need to process: {}".format(len(imls)))
    
    for i, imname in enumerate(imls):
        logging.info("{}. {}".format(i+1, imname))
        img = sitk.ReadImage(os.path.join(args.imdir, imname))
        
        if args.maskdir is not None:
            spaces = img.GetSpacing()
            imname = re.sub(r"_0000$", "", imname)
            mask = sitk.ReadImage(os.path.join(args.maskdir, imname))
            ma = sitk.GetArrayFromImage(mask)
            bin_mask = ma.copy()
            bin_mask[bin_mask != 0] = 1
            max_clump = get_largest_connectivity(bin_mask)
            cropped_im, cropped_ma = crop_heart_bbx(im, max_clump, pad=(5,5,5))
            resampled_im = maskout_non_heart(cropped_im, cropped_ma)
            resampled_im = norm_min_max(resampled_im)
            img = sitk.GetImageFromArray(resampled_im)
            img.SetSpacing(spaces)
        
        ori_im = sitk.GetArrayFromImage(img)
        img = resample3d(img, input_shape[::-1])
        im = sitk.GetArrayFromImage(img)
        im = norm_min_max(im)
        im = np.expand_dims(im, axis=3)
        im = np.expand_dims(im, axis=0)
        
        pred = model.predict(tf.convert_to_tensor(im))
        class_idx = np.argmax(pred)
        cam = grad_model.compute_heatmap(image=im,
                                         classIdx=class_idx,
                                         upsample_size=ori_im.shape)
        colored_cam, colored_overlay = overlay3d(ori_im, cam)
        
        if class_idx == 0:
            savedir = os.path.join(outdir, "negatives", imname)
        else:
            savedir = os.path.join(outdir, "positives", imname)
        os.makedirs(savedir, exist_ok=True)
            
        # save cam
        tifffile.imwrite(os.path.join(savedir, "cam.tif"), colored_cam)
        # save overlay cam
        tifffile.imwrite(os.path.join(savedir, "overlay.tif"), colored_overlay)
        # Save im as tif
        tifffile.imwrite(os.path.join(savedir, "image.tif"), np.uint8(ori_im*255))
        
        # Parallel
        axial_parallel = generate_parellel(ori_im, colored_overlay, view="axial")
        tifffile.imwrite(os.path.join(savedir, "parallel_axial.tif"), axial_parallel)
        coronal_parallel = generate_parellel(ori_im, colored_overlay, view="coronal")
        tifffile.imwrite(os.path.join(savedir, "parallel_coronal.tif"), coronal_parallel)
        sagittal_parallel = generate_parellel(ori_im, colored_overlay, view="sagittal")
        tifffile.imwrite(os.path.join(savedir, "parallel_sagittal.tif"), sagittal_parallel)
