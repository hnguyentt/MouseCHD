import json, os, shutil
import logging
import datetime
import pandas as pd
import argparse

from mousechd.datasets.utils import get_latest_meta
from mousechd.datasets.preprocess import DataSplitter
from mousechd.utils.tools import set_logger

def add_args(parser):
    parser.add_argument("-datadir", help="Path to the data directory")
    parser.add_argument("-metafile", help="Path to the meta file", default=None)
    parser.add_argument("-sep", help="Separator in the meta file", default=",")
    parser.add_argument("-seed", help="Random seed", type=int, default=42)
    parser.add_argument("-nnunet_dir", help="Path to the nnUNet directory of the task")
    parser.add_argument("-continue_from", help="Continue from other task, ex: Task103_MouseHeartImagine", default=None)
    parser.add_argument("-test_size", type=float, help="Test size", default=0.2)
    parser.add_argument("-save_test", type=int, choices=[0,1], help="Save test data", default=0)
    
    return parser

def main(args):
    NOW = datetime.datetime.now().strftime('%Y%m%d')
    # Setting logger
    set_logger(os.path.join(args.datadir, "{}.log".format(os.path.basename(args.nnunet_dir))))
    
    logging.info("Starting nnUNet data preparation at {}".format(NOW))
    logging.info("Arguments: {}".format(args.__dict__))
    
    if args.metafile is None:
        metafile = get_latest_meta(args.datadir, "metadata")
    else:
        metafile = args.metafile
        
    df = pd.read_csv(metafile, sep=args.sep)
    
    os.makedirs(os.path.join(args.nnunet_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(args.nnunet_dir, "labelsTr"), exist_ok=True)
    
    # Split test hearts into subset for faster testing
    if args.save_test:
        all_hearts = [x for x in os.listdir(os.path.join(args.datadir, "images"))
                  if x.endswith(".nii.gz") and not x.startswith(".")]
        all_hearts = [x.split(".")[0] for x in all_hearts]
        all_hearts = [x for x in all_hearts if x not in ["N_261h_0000", "NH_229m_0000"]] # error images
        test_hearts_df = df[df["Stage"].isin(["P0", "E18.5"]) & df["heart_name"].isin(all_hearts)]
        test_hearts = test_hearts_df["heart_name"].to_list()
        logging.info("Total number of images for testing: {}".format(len(test_hearts)))
        
        for i in range(len(test_hearts)//10+1):
            if len(test_hearts[10*i:(10*i+10)]) > 0:
                logging.info("Saving test subset {}".format(i+1))
                os.makedirs(os.path.join(args.datadir, "nnUNetTs", "subset{:02d}".format(i+1)), exist_ok=True)
        
            for heart_name in test_hearts[10*i:(10*i+10)]: 
                shutil.copy(os.path.join(args.datadir, "images", heart_name + ".nii.gz"),
                            os.path.join(args.datadir, "nnUNetTs", "subset{:02d}".format(i+1), "{}_0000.nii.gz".format(heart_name)))
    
    try:
        task = json.load(open(os.path.join(args.datadir, "{}.json".format(os.path.basename(args.nnunet_dir)))))
        X_train, X_test = task["X_train"], task["X_test"]
        logging.info("Load previous process")
    except FileNotFoundError:
        if args.continue_from is not None:
            old_task = json.load(open(os.path.join(args.datadir, "{}.json".format(args.continue_from))))
            old_X_train, old_X_test = old_task["X_train"], old_task["X_test"]
            df = df[~df["heart_name"].isin(old_X_train + old_X_test)]
        else:
            old_X_train, old_X_test = [], []
           
        X_train, X_test = DataSplitter(df, os.path.join(args.datadir, "heart-masks")).split(
            test_size=args.test_size, val_size=0.0, stratify=True, seed=args.seed
        )
        
        X_train = X_train["heart_name"].to_list() + old_X_train
        X_test = X_test["heart_name"].to_list() + old_X_test
        json.dump({"X_train": X_train, "X_test": X_test}, 
                  open(os.path.join(args.datadir, "{}.json".format(os.path.basename(args.nnunet_dir))), "w"), indent=4)
    
    dataset = {
        "name": "MouseHeartSegImagine",
        "description": "Whole mouse heart segmentation with data from Imagine",
        "tensorImageSize": "3D",
        "reference": "Institut Imagine",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 14/02/2021",
        "modality": {"0": "microCT"},
        "labels": {
            "0": "background",
            "1": "heart"
        },
        "numTraining": len(X_train),
        "numTest": len(X_test),
        "training": [],
        "test": []
    }
    logging.info("Train: {}".format(len(X_train)))
    logging.info("Test: {}".format(len(X_test)))
    
    logging.info("Preparing training data")
    for heart_name in list(X_train):
        shutil.copy(os.path.join(args.datadir, "images", heart_name + "_0000.nii.gz"),
                    os.path.join(args.nnunet_dir, "imagesTr", "{}_0000.nii.gz".format(heart_name)))
        
        shutil.copy(os.path.join(args.datadir, "heart-masks", heart_name + ".nii.gz"),
                    os.path.join(args.nnunet_dir, "labelsTr", "{}.nii.gz".format(heart_name)))
        
        dataset["training"].append({"image": "./imagesTs/{}.nii.gz".format(heart_name),
                                    "label": "./labelsTr/{}.nii.gz".format(heart_name)})
        
    for heart_name in list(X_test):
        dataset["test"].append("./imagesTs/{}.nii.gz".format(heart_name))
        
    # Save dataset
    with open(os.path.join(args.nnunet_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for nnUNet')
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
