import os
import argparse

from mousechd.datasets.resample import resample_folder
from mousechd.utils.tools import set_logger

def add_args(parser):
    parser.add_argument("-imdir", type=str, help="Input image directory")
    parser.add_argument("-maskdir", type=str, help="Input mask directory")
    parser.add_argument("-outdir", type=str, help="Output directory")
    parser.add_argument("-metafile", type=str, help="Input meta file", default=None)
    parser.add_argument("-sep", type=str, help="Input meta file", default=",")
    parser.add_argument("-save_images", type=int, choices=[0,1], help="save images?", default=0)
    parser.add_argument("-logfile", type=str, help="path to logfile", default=None)

    return parser
    
def main(args): 
    # Logging
    os.makedirs(name=args.outdir, exist_ok=True)
    if args.logfile is not None:
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        set_logger(args.logfile)
    else:
        set_logger(os.path.join(args.outdir, "resample.log"))
        
    resample_folder(imdir=args.imdir,
                    maskdir=args.maskdir,
                    outdir=args.outdir,
                    metafile=args.metafile,
                    meta_sep=args.sep,
                    save_images=bool(args.save_images))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample mouse heart dataset')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)