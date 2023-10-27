import os
import argparse

from mousechd.segmentation.segment import segment_from_folder


def add_args(parser):
    parser.add_argument("-indir", type=str, help="Input directory")
    parser.add_argument("-outdir", type=str, help="Output directory")
    parser.add_argument("-fold", type=int, help="Fold to predict. Default: None, meaning ensemble all folds", default=None)
    
    return parser


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    segment_from_folder(args.indir, args.outdir, folds=args.fold)
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Heart segmentation")
    parser = add_args(parser)
    args = parser.parse_args()
    main()