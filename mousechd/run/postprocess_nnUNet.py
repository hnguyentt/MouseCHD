import os, re
from tqdm import tqdm
import argparse
import tifffile
import SimpleITK as sitk

from mousechd.datasets.utils import get_largest_connectivity, EXCLUDE_IDS
    
def add_args(parser):
    parser.add_argument('-indir', type=str, help='Input directory')
    parser.add_argument('-outdir', type=str, help='Output directory')

    return parser

def main(args):
    
    os.makedirs(name=args.outdir, exist_ok=True)
    
    processed = [re.sub(r".tif$", "", x) for x in os.listdir(args.outdir)
                 if (not x.startswith(".")) and x.endswith(".tif")]
    im_ls = [re.sub(r".nii.gz$", "", x) for x in os.listdir(args.indir)
             if (not x.startswith(".")) and x.endswith(".nii.gz")]
    im_ls = [x for x in im_ls if x not in EXCLUDE_IDS]
    print("Total: {}".format(len(im_ls)))
    print("Post-processed: {}".format(len(processed)))
    im_ls = [x for x in im_ls if x not in processed]
    print("Need to post-process: {}".format(len(im_ls)))
    for imname in tqdm(im_ls):
        mask = sitk.ReadImage(os.path.join(args.indir, imname))
        mask = sitk.GetArrayFromImage(mask)
        bin_mask = mask.copy()
        bin_mask[bin_mask != 0] = 1
        max_clump = get_largest_connectivity(bin_mask)
        mask[max_clump==0] = 0
        
        tifffile.imwrite(os.path.join(args.outdir, f"{imname}.tif"), mask)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Postprocess NNUNet output')
    parser = add_args(parser)
    args = parser.parse_args()
    
    main(args)
