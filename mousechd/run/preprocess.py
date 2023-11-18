"""
PREPROCESSING DATA FROM RAW DATA IN DICOM FORMAT.
It is recommended that your data is structured in the following way:
DATABASE # your database name
└── raw # raw folder to store raw data
    ├── NameOfDataset1 # name of dataset
    │   ├── images_20200206 # folder to store images recieved on 20200206 [YYYYMMDD]
    │   ├── masks_20210115 # folder to store masks recieved on 20210115 [YYYYMMDD]
    │   ├── masks_20210708 # folder to store masks recieved on 20210708 [YYYYMMDD]
    │   └── metadata_20210703.csv # metadata file received on 20210703 [YYYYMMDD]
    └── NameOfDataset2 # name of another dataset
        └── images_20201010
        ......
"""

import argparse
import os

from mousechd.datasets.preprocess import Preprocess, ALL_MASKTYPES, ALL_IMAGE_FORMATS
from mousechd.utils.tools import set_logger

def add_args(parser):
    parser.add_argument("-database", type=str, help="database name")
    parser.add_argument("-imdir", type=str, help="image directory (relative to database)")
    parser.add_argument("-im_format", type=str, help="file format of image", choices=ALL_IMAGE_FORMATS, default="DICOM")
    parser.add_argument("-maskdir", type=str, help="mask directory (relative to database)", default=None)
    parser.add_argument("-masktype", type=str, choices=ALL_MASKTYPES, default="NIFTI")
    parser.add_argument("-metafile", type=str, help="path to metadata file", default=None)
    parser.add_argument("-sep", type=str, help="separator in metadata file", default=",")
    parser.add_argument("-outdir", type=str, help="output directory (relative to database)")
    parser.add_argument("-logfile", type=str, help="path to logfile", default=None)
    parser.add_argument("-kwargs", type=str, help="optional information such as 'orientation', 'spacing'",
                        default="{'orientation': 'SAR', 'spacing': (0.02, 0.02, 0.02)}")
    
    return parser

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    if args.logfile is not None:
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        set_logger(args.logfile)
    else:
        set_logger(os.path.join(args.outdir, "preprocess.log"))
    
    Preprocess(database=args.database, 
               imdir=args.imdir,
               outdir=args.outdir, 
               im_format=args.im_format,
               maskdir=args.maskdir, 
               masktype=args.masktype,
               metafile=args.metafile,
               sep=args.sep,
               **eval(args.kwargs)).preprocess()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw data from DICOM format.')
    parser = add_args(parser)
    args = parser.parse_args()
    main()
