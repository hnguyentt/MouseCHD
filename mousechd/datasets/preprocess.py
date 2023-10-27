"""
Preprocessing images for mouse heart scans 
"""
import logging
import pandas as pd
import pydicom
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from skimage import exposure
import os, re
from pathlib import Path

from .utils import (load_tif3d, 
                    load_slices, 
                    load_nifti,
                    anyview2LPS, 
                    dicom2nii, mirror, 
                    get_view_sitk,
                    nrrd2nii,
                    get_origin_heartname,
                    get_stats_df,
                    get_stats_kfolds,
                    add_prefix_kfolds,
                    save_fold_labels,
                    make_isotropic
                    )
from .utils import RESAMPLE_METHODS, INTEREST_FIELDS
from ..utils.tools import txt2list

ALL_MASKTYPES = ["TIF2d", "TIF3d", "NIFTI"]
ALL_IMAGE_FORMATS = ["DICOM", "NRRD", "NIFTI"]


class Preprocess(object):
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
    def __init__(self, 
                 database, 
                 imdir, 
                 outdir,
                 im_format="DICOM",
                 metafile=None, 
                 maskdir=None,
                 masktype="NIFTI", 
                 sep=";",
                 **kwargs
                 ):
        
        assert masktype in ALL_MASKTYPES, f"Mask type must be in {ALL_MASKTYPES}"
        assert im_format in ALL_IMAGE_FORMATS, f'im_format must be in {ALL_IMAGE_FORMATS}'
        self.database = database
        self.imdir = os.path.join(database, imdir)
        self.metafile = metafile
        self.outdir = outdir
        self.im_format = im_format
        self.maskdir = None if maskdir is None else os.path.join(database, maskdir)
        self.masktype = masktype
        self.sep = sep
        self.kwargs = kwargs
        
        if masktype == "TIF2d":
            self.load_mask = load_slices
        elif masktype == "TIF3d":
            self.load_mask = load_tif3d
        else:
            self.load_mask = load_nifti
            
        
        os.makedirs(os.path.join(self.outdir, "images"), exist_ok=True)
            
        os.makedirs(os.path.join(self.outdir, "images"), exist_ok=True)
        if maskdir is not None:
            os.makedirs(os.path.join(self.outdir, "heart-masks"), exist_ok=True)
        
        # Logging
        logging.info("\n\n")
        logging.info("Log file is saved in {}".format(os.path.join(self.outdir, "preprocess.log")))
        logging.info("="*50)
        logging.info("PREPROCESSING STARTS")
        logging.info("-database: {}".format(self.database))
        logging.info("-imdir: {}".format(self.imdir))
        logging.info("-outdir: {}".format(self.outdir))
        logging.info("-im_format: {}".format(self.im_format))
        logging.info("-metafile: {}".format(self.metafile))
        logging.info("-maskdir: {}".format(self.maskdir))
        logging.info("-masktype: {}".format(self.masktype))
        logging.info("-sep: {}".format(self.sep))
        
            
    def load_progress(self):
        """Load progress of preprocessing

        Returns:
            (dataframe): processing dataframe
        """
        logging.info("ASSEMBLE ALL HEARTS")
        all_hearts = sorted([x for x in os.listdir(self.imdir) if not x.startswith(".")])
        all_hearts = [str(Path(self.imdir).relative_to(self.database)) + os.sep + x for x in all_hearts]
        logging.info("{} hearts".format(len(all_hearts)))
        
        if self.kwargs.get("exclude", None) is not None:
            exclude_ls = txt2list(self.kwargs.get("exclude"))
            all_hearts = [x for x in all_hearts if x not in exclude_ls]
            logging.info("Excluing hearts in {}. Need to process: {}".format(self.kwargs.get("exclude"), len(all_hearts)))
            
        if self.kwargs.get("batch", None) is not None:
            batch = int(self.kwargs.get("batch"))
            all_hearts = all_hearts[:batch]
            logging.info("Process {} first hearts".format((batch)))
        
        try:
            df = pd.read_csv(os.path.join(self.outdir, "processed.csv"))
            unprocessed = [x for x in all_hearts if x not in df["folder"].to_list()]
            
            if len(unprocessed) > 0:
                unprocessed_df = pd.DataFrame({"folder": unprocessed})
                df = pd.concat([df, unprocessed_df], ignore_index=True)
                
            logging.info("LOAD PREVIOUS PROGRESS")
            
        except FileNotFoundError:
            df = pd.DataFrame({"folder": all_hearts})
            df["heart_name"] = None
            df["origin_view"] = None
            df["size"] = None
            df["spacing"] = None
            df["heartmask"] = None
            df.to_csv(os.path.join(self.outdir, "processed.csv"), index=False)
            
        return df
    
    def preprocess(self):
        df = self.load_progress()
        logging.info("Total: {}".format(len(df)))
        unprocessed = df[df["heart_name"].isna()]["folder"].to_list()
        logging.info("Need to process: {}".format(len(unprocessed)))
        
        df.set_index("folder", inplace=True)
        
        for i, x in enumerate(unprocessed):
            logging.info("{}. {}".format(i+1, x))
            
            if self.im_format == "DICOM": #TODO: improve speed by define image reader beforehand
                first_file = next(os.path.join(self.database, x, f)
                                    for f in os.listdir(os.path.join(self.database, x)) if not f.startswith("."))
                dicom_data = pydicom.read_file(first_file)
                mouse = str(dicom_data.get("PatientName")).replace(" ","") + "_0000"
                if mouse in df["heart_name"].to_list():
                    logging.info("Already processed")
                    df.loc[x, "heart_name"] = "duplicated"
                    df.to_csv(os.path.join(self.outdir, "processed.csv"))
                    continue
                
                img = dicom2nii(os.path.join(self.database, x))
                
            elif self.im_format == "NIFTI":
                mouse = re.sub(r".nii.gz$", "", x.split(os.sep)[-1]) + "_0000"
                img = sitk.ReadImage(os.path.join(self.database, x))
            else:
                mouse = "{}_0000".format(re.sub(r".nrrd$", "", x.split(os.sep)[-1]))
                img = nrrd2nii(os.path.join(self.database, x), 
                               orientation=self.kwargs.get('orientation', 'SAR'),
                               spacing=self.kwargs.get('spacing', (0.02, 0.02, 0.02)))
                
            logging.info("Mouse: {}".format(mouse))
            
            processed_img = anyview2LPS(img)
            processed_img = make_isotropic(processed_img)
            
            sitk.WriteImage(processed_img, 
                            os.path.join(self.outdir, "images", "{}.nii.gz".format(mouse))) # 
            df.loc[x, "heart_name"] = re.sub(r"_0000$", "", mouse)
            df.loc[x, "origin_view"] = get_view_sitk(img)
            df.loc[x, "size"] = str(processed_img.GetSize())
            df.loc[x, "spacing"] = str(processed_img.GetSpacing())
            del processed_img
            
            if self.maskdir is not None:
                try:
                    mask_arr = self.load_mask(os.path.join(self.maskdir, df.loc[x,"heart_name"]))
                    mask = mirror(img=img, seg=mask_arr)
                    processed_mask = anyview2LPS(mask)
                    processed_mask = make_isotropic(processed_mask)
                    sitk.WriteImage(processed_mask, 
                                    os.path.join(self.outdir, 
                                                 "heart-masks", 
                                                 "{}.nii.gz".format(re.sub(r"_0000$", "", mouse))))
                    df.loc[x, "heartmask"] = True
                except FileNotFoundError:
                    pass
            del img
                
            df.to_csv(os.path.join(self.outdir, "processed.csv"))
            
        df.reset_index(inplace=True, drop=False)
        df[INTEREST_FIELDS].to_csv(os.path.join(self.outdir, "processed.csv"), index=False)
        
        logging.info("PROCESSING IMAGES DONE!")
        
        if self.maskdir is not None:
            mask_ls = [x for x in os.listdir(self.maskdir) if not x.startswith(".")]
            mask_ls = [x.split(".")[0] for x in mask_ls]
            processed_masks = df[~df["heartmask"].isna()]["heart_name"].to_list()
            unprocessed_masks = [x for x in mask_ls if x not in processed_masks]
            
            if len(unprocessed_masks) > 0:
                logging.info("Unprocessed masks: {}".format(len(unprocessed_masks)))
                logging.info(unprocessed_masks)
                
            assert all(x in df.heart_name.to_list() for x in unprocessed_masks), "Some masks are present in the absence of images"
            
            df.set_index("heart_name", inplace=True)
            
            for i, x in enumerate(unprocessed_masks):
                
                img = dicom2nii(os.path.join(self.database, df.loc[x, "folder"]))
                mask_arr = self.load_mask(os.path.join(self.maskdir, x))
                mask = mirror(img=img, seg=mask_arr)
                processed_mask = anyview2LPS(mask)
                processed_mask = make_isotropic(processed_mask)
                sitk.WriteImage(processed_mask, 
                                os.path.join(self.outdir, "heart-masks", "{}.nii.gz".format(x)))
                df.loc[x, "heartmask"] = True
                
                df.to_csv(os.path.join(self.outdir, "processed.csv"))
            
            df.reset_index(inplace=True, drop=False)
            df[INTEREST_FIELDS].to_csv(os.path.join(self.outdir, "processed.csv"), index=False)
            
        if self.metafile is not None:
            logging.info("PROCESS METADATA")
            df = df[df["heart_name"] != "duplicated"]
            df["heart_name"] = df["heart_name"].astype(str)
            meta = pd.read_csv(self.metafile, sep=self.sep, converters={"heart_name": str})
            meta["heart_name"] = str(Path(self.imdir).relative_to(self.database)) + os.sep + meta["heart_name"]
            meta["heart_name"] = meta["heart_name"].map(
                df.set_index("folder").to_dict()["heart_name"]
                ).fillna(meta["heart_name"].str.replace(str(Path(self.imdir).relative_to(self.database)) + os.sep, ""))
            
            meta.drop_duplicates(subset=["heart_name"], inplace=True)
            processed_meta = pd.DataFrame({"heart_name": list(set(df.heart_name.to_list()))})
            processed_meta = processed_meta.merge(meta, how="left", on="heart_name")
            
            processed_meta.to_csv(os.path.join(self.outdir, os.path.basename(self.metafile)), index=False)
            
            
class DataSplitter(object):
    """
    Split data into train, val and test sets
    """
    def __init__(self, df):
        
        if "label" not in df.columns:    
            df["label"] = 1 - df["Normal heart"]
            
        df["stratify"] = df["label"].astype(str) +  "_" + df["Stage"].astype(str)
        if sum(df.stratify.value_counts() <=1) > 0:
            logging.info("WARNING: Some images are not stratified by label and stage")
            logging.info("Stratify by label only")
            df["stratify"] = df["Normal heart"].astype(str)
        
        self.df = df
        
        
    def split(self, test_size, val_size=0.0, stratify=True, seed=42, df=None):
        if df is None:
            df = self.df
        
        if stratify:
            X_train, X_test, _, _ = train_test_split(df["heart_name"], 
                                                     df["stratify"], 
                                                     stratify=df["stratify"],
                                                     test_size=test_size, 
                                                     random_state=seed)
        else:
            X_train, X_test, _, _ = train_test_split(df["heart_name"], 
                                                     df["stratify"], 
                                                     test_size=test_size, 
                                                     random_state=seed)
            
        if val_size > 0:
            df = df[df["heart_name"].isin(X_train)]
            if stratify:
                X_train, X_val, _, _ = train_test_split(df["heart_name"], 
                                                        df["stratify"], 
                                                        stratify=df["stratify"],
                                                        test_size=val_size, 
                                                        random_state=seed)
            else:
                X_train, X_val, _, _ = train_test_split(df["heart_name"], 
                                                        df["stratify"], 
                                                        test_size=val_size, 
                                                        random_state=seed)
                
            return (df[df["heart_name"].isin(X_train.to_list())][["heart_name", "label"]].reset_index(drop=True),
                    df[df["heart_name"].isin(X_val.to_list())][["heart_name", "label"]].reset_index(drop=True),
                    df[df["heart_name"].isin(X_test.to_list())][["heart_name", "label"]].reset_index(drop=True))
        else:
            return (df[df["heart_name"].isin(X_train.to_list())][["heart_name", "label"]].reset_index(drop=True),
                    df[df["heart_name"].isin(X_test.to_list())][["heart_name", "label"]].reset_index(drop=True))
    
         
    def split_k_folds(self, val_size=0.0, k=5, seed=42, df=None):
        if df is None:
            df = self.df
            
        results = []
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        for train_index, test_index in skf.split(df.heart_name, df.stratify):
            res = {}
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            if val_size > 0:
                train_df, val_df = self.split(test_size=val_size, stratify=True, seed=seed, df=train_df)
                
                res["val"] = val_df[["heart_name", "label"]].reset_index(drop=True)
                
            res["train"] = train_df[["heart_name", "label"]].reset_index(drop=True)
            res["test"] = test_df[["heart_name", "label"]].reset_index(drop=True)
            
            results.append(res)
            
        return results
    

def x5_df(df):
    df_cp = df.copy()
    df_cp.reset_index(inplace=True, drop=True)
    imls = []
    labels = []
    
    for i, heart_name in enumerate(df_cp["heart_name"].values):
        imls += ["{}_{:02d}".format(heart_name, x) for x in range(1,6)]
        labels += [df_cp.loc[i,"label"]]*5
    
    return pd.DataFrame({"heart_name": imls, "label": labels})


def x5_kfolds(base_data):
    kfolds = []
    for i, fold in enumerate(base_data):
        train_df = x5_df(fold["train"])
        test_df = x5_df(fold["test"])
        val_df = x5_df(fold["val"])
        kfolds.append({"train": train_df, "test": test_df, "val": val_df})
    
    return kfolds


def split_k_folds(meta_df, num_folds, val_size, outdir, seed=42):
    try:
        splitted_data = []
        for i in range(num_folds):
            train_df = pd.read_csv(os.path.join(outdir, "base", f"{num_folds}folds", f"F{i+1}", "train.csv"))
            val_df = pd.read_csv(os.path.join(outdir, "base", f"{num_folds}folds", f"F{i+1}", "val.csv"))
            test_df = pd.read_csv(os.path.join(outdir, "base", f"{num_folds}folds", f"F{i+1}", "test.csv"))
            splitted_data.append({"train": train_df, "val": val_df, "test": test_df})
            logging.info("Load previous base splits")
    except FileNotFoundError:
        splitter = DataSplitter(df=meta_df)
        splitted_data = splitter.split_k_folds(val_size=val_size, k=num_folds, seed=seed)
        save_fold_labels(save_dir=os.path.join(outdir, "base", "{}folds".format(num_folds)), kfolds=splitted_data)
        
    logging.info("Basic labels: ")
    get_stats_kfolds(splitted_data, meta_df)
    
    dfs = x5_kfolds(base_data=splitted_data)
    save_fold_labels(save_dir=os.path.join(outdir, "x5", "{}folds".format(num_folds)), kfolds=dfs)    
    logging.info("\n********** x5 **********")
    get_stats_kfolds(dfs, meta_df)
    
    # Merge base and x5
    merged_dfs = []
    pref_bases = add_prefix_kfolds(splitted_data, "images{}".format(os.sep))
    pref_x5 = add_prefix_kfolds(dfs, "images_x5{}".format(os.sep))
    for i in range(num_folds):
        merged_dfs.append({
            "train": pd.concat([pref_bases[i]["train"], pref_x5[i]["train"]], ignore_index=True),
            "test": pd.concat([pref_bases[i]["test"], pref_x5[i]["test"]], ignore_index=True),
            "val": pd.concat([pref_bases[i]["val"], pref_x5[i]["val"]], ignore_index=True)
        })
    save_fold_labels(os.path.join(outdir, "x5_base", "{}folds".format(num_folds)), merged_dfs)
    logging.info("\n********** x5 + base **********")
    get_stats_kfolds(merged_dfs, meta_df)
    

def merge_base_x5_labels(df, df_x5):
    """Merge base and x5 labels

    Args:
        df (pd.DataFrame): base dataframe
        df_x5 (pd.DataFrame): x5 dataframe
        prefix (str): prefix

    Returns:
        _type_: _description_
    """
    df_cp = df.copy()
    df_x5_cp = df_x5.copy()
    df_cp["heart_name"] = "images" + os.sep + df_cp["heart_name"]
    df_x5_cp["heart_name"] = "images_x5" + os.sep + df_x5_cp["heart_name"]
    
    return pd.concat([df_cp, df_x5_cp], ignore_index=True)

def split_data(meta_df, outdir, val_size, seed=42):
    try:
        train_df = pd.read_csv(os.path.join(outdir, "base", "1fold", "train.csv"))
        val_df = pd.read_csv(os.path.join(outdir, "base", "1fold", "val.csv"))
        logging.info("Load base data")
    except FileNotFoundError:
        os.makedirs(os.path.join(outdir, "base", "1fold"), exist_ok=True)
        splitter = DataSplitter(df=meta_df)
        train_df, val_df = splitter.split(test_size=val_size,
                                          val_size=0.,
                                          stratify=True,
                                          seed=seed)
        train_df.to_csv(os.path.join(outdir, "base", "1fold", "train.csv"), index=False)
        val_df.to_csv(os.path.join(outdir, "base", "1fold", "val.csv"), index=False)
    
    logging.info("********** base **********")
    logging.info("===>TRAIN")
    get_stats_df(train_df, meta_df)
    logging.info("===>VAL")
    get_stats_df(val_df, meta_df)
    
    # x5
    train_df_x5 = x5_df(train_df)
    val_df_x5 = x5_df(val_df)
    os.makedirs(os.path.join(outdir, "x5", "1fold"), exist_ok=True)
    train_df_x5.to_csv(os.path.join(outdir, "x5", "1fold", "train.csv"), index=False)
    val_df_x5.to_csv(os.path.join(outdir, "x5", "1fold", "val.csv"), index=False)
    
    logging.info("********** x5 **********")
    logging.info("===>TRAIN")
    get_stats_df(train_df_x5, meta_df)
    logging.info("===>VAL")
    get_stats_df(val_df_x5, meta_df)
    
    # Merge base and x5
    merged_train = merge_base_x5_labels(df=train_df, df_x5=train_df_x5)
    merged_val = merge_base_x5_labels(df=val_df, df_x5=val_df_x5)
    os.makedirs(os.path.join(outdir, "x5_base", "1fold"), exist_ok=True)
    merged_train.to_csv(os.path.join(outdir, "x5_base", "1fold", "train.csv"), index=False)
    merged_val.to_csv(os.path.join(outdir, "x5_base", "1fold", "val.csv"), index=False)
    
    logging.info("********** x5 + base **********")
    logging.info("===>TRAIN")
    get_stats_df(merged_train, meta_df)
    logging.info("===>VAL")
    get_stats_df(merged_val, meta_df)