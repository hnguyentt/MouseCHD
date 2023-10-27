import tifffile
import os
import argparse

from mousechd.datasets.utils import get_largest_connectivity, EXCLUDE_IDS
    
def add_args(parser):
    parser.add_argument('-indir', type=str, help='Input directory')
    parser.add_argument('-outdir', type=str, help='Output directory')

    return parser

def main(args):
    
    os.makedirs(name=args.outdir, exist_ok=True)
    
    processed = [x for x in os.listdir(args.outdir)
                 if (not x.startswith(".")) and x.endswith("tif")]
    im_ls = [x for x in os.listdir(args.indir)
             if (not x.startswith(".")) and x.endswith("tif")]
    im_ls = [x for x in im_ls if x.split(".")[0] not in EXCLUDE_IDS]
    im_ls = [x for x in im_ls if x not in processed]
    print(len(im_ls))
    for imname in im_ls:
        print(imname)
        mask = tifffile.imread(os.path.join(args.indir, imname))
        bin_mask = mask.copy()
        bin_mask[bin_mask != 0] = 1
        max_clump = get_largest_connectivity(bin_mask)
        mask[max_clump==0] = 0
        print("Max clump: {}, {}, {}".format(max_clump.sum(),
                                             max_clump.min(),
                                             max_clump.max()))
        
        tifffile.imwrite(os.path.join(args.outdir, imname), mask)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Postprocess NNUNet output')
    parser = add_args(parser)
    args = parser.parse_args()
    
    main(args)
