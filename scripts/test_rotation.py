import os, re, logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

from mousechd.datasets.utils import (resample3d, 
                                     rotate3d,
                                     norm_min_max)

from mousechd.classifier.utils import calculate_metrics, eval_clf
from mousechd.classifier.evaluate import summarize_results
from mousechd.classifier.utils import (download_clf_models, 
                                       CLF_DIR)

class MouseCHDEvalGen(tf.keras.utils.Sequence):
    def __init__(self,
                 imdir,
                 target_size,
                 maskdir=None,
                 filenames=None,
                 batch_size=8,
                 labels=None,
                 stage="eval",
                 theta_x=0):
        
        assert stage in ["eval", "test"], "stage must be in ['eval', 'test']"
        
        self.imdir = imdir
        self.maskdir = maskdir
        if filenames is None:
            if maskdir is None:
                filenames = [re.sub(r".nii.gz$", "", x) for x in os.listdir(imdir)]
            else:
                filenames = [re.sub(r".nii.gz$", "", x) for x in os.listdir(maskdir)]
        self.filenames = filenames
        self.batch_size = batch_size
        self.target_size = target_size
        self.list_IDs = [i for i in range(len(filenames))]
        self.theta_x = theta_x
        
        if labels is None:
            labels = [None] * len(filenames)
        self.labels = labels
        
        self.stage = stage
        
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.filenames)/float(self.batch_size)))
    

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        return self.__datagen(list_ids_tmp)
        
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
    
    
    def __datagen(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = []
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            img = sitk.ReadImage(os.path.join(self.imdir, self.filenames[ID]))
            img = rotate3d(img, theta_z=self.theta_x)
            img = resample3d(img, self.target_size[:3][::-1])
            im = sitk.GetArrayFromImage(img)
            im = norm_min_max(im)
            im = np.expand_dims(im, axis=3)
            
            X[i,:] = im
            y.append(self.labels[ID])
            files.append(self.filenames[ID])
            
        return X, y, files
    
    
def predict_folder(model,
                   imdir,
                   maskdir,
                   target_size,
                   label_df=None,
                   stage="test",
                   batch_size=8,
                   save=None,
                   grouped_result=False,
                   theta_x=0):
    assert stage in ["test", "eval"], "stage must be in ['test', 'eval']"
    
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
    
    try:
        df = pd.read_csv(save)
    except:
        df = pd.DataFrame(columns=["heart_name",
                                    "label",
                                    "prob",
                                    "pred"])
    
    if label_df is None:
        filenames = [x for x in os.listdir(maskdir) if (not x.startswith(".") and x.endswith(".nii.gz"))]
        filenames = [re.sub(r".nii.gz$", "", x) for x in filenames]
        labels = [None]*len(filenames)
    else:
        filenames = label_df["heart_name"].values
        labels = label_df["label"].values
    
    # Check already predicted cases    
    logging.info("Number of cases: {}".format(len(filenames)))
    label_dict = dict(zip(filenames, labels))
    filenames = [x for x in filenames if x not in df["heart_name"].values]
    labels = [label_dict[x] for x in filenames]
    logging.info("Number of case need to predict: {}".format(len(filenames)))
    
    # prevent at the epoch end problem
    break_idx = int((len(filenames)//batch_size)*batch_size)
    
    test_gen = MouseCHDEvalGen(imdir=imdir,
                               target_size=target_size,
                               maskdir=maskdir,
                               filenames=filenames[:break_idx],
                               batch_size=batch_size,
                               labels=labels[:break_idx],
                               stage=stage,
                               theta_x=theta_x)
    
    df = eval_clf(model=model,
                  datagen=test_gen,
                  batch_size=batch_size,
                  stage=stage,
                  df=df,
                  save=save)
            
    # Predict for remained cases
    unprocessed_fn = filenames[break_idx:]
    if len(unprocessed_fn) > 0:
        test_gen = MouseCHDEvalGen(imdir=imdir,
                                   target_size=target_size,
                                   maskdir=maskdir,
                                   filenames=unprocessed_fn,
                                   batch_size=1,
                                   labels=labels[break_idx:],
                                   stage=stage,
                                   theta_x=theta_x
                                   )
        df = eval_clf(model=model,
                      datagen=test_gen,
                      batch_size=1,
                      stage=stage,
                      df=df,
                      save=save)
       
    logging.info("Final metrics: {}".format(calculate_metrics(df["prob"].values.astype(float), 
                                                              df["label"].values.astype(float))))
            
    if grouped_result:
        df["heart_name"] = df["heart_name"].str.replace("_\d+$", "")
        group_df = df.groupby(by="heart_name").agg("mean")
        group_df["pred"] = (group_df["prob"] > 0.5)*1
        logging.info("Grouped results: {}".format(calculate_metrics(group_df["prob"].values.astype(float),
                                                                    group_df["label"].values.astype(float))))
    
    return df
    
    
def main(args):
    from mousechd.classifier.models import load_MouseCHD_model
    from mousechd.utils.tools import set_logger

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.outdir is None:
        if args.fold is None:
            sum_df_path = os.path.join(args.model_dir, "results", "rotation", "summary.csv")
        else:
            sum_df_path = os.path.join(args.model_dir, args.fold, "results", "rotation", "summary.csv")
    else:
        sum_df_path = os.path.join(args.outdir, "summary.csv")
        
    # Summary dataframe
    try:
        sum_df = pd.read_csv(sum_df_path)
    except FileNotFoundError:
        sum_df = pd.DataFrame(columns=["ckpt", "testfile",
                                       "acc", "bal_acc",
                                       "sens", "spec", "auc"])
        
    # Download models if necessary
    if args.model_dir is None:
        download_clf_models()
        conf_path = os.path.join(CLF_DIR, "configs.json")
    else:
        if args.fold is None:
            conf_path = os.path.join(args.model_dir, "configs.json")
        else:
            conf_path = os.path.join(args.model_dir, args.fold, "configs.json")
    
    # List of checkpoints needing evaluation    
    ckpt = "best_model.hdf5"
    
    if args.outdir is None:
        outdir = os.path.join(os.path.dirname(sum_df_path), ckpt)
    else:
        outdir = args.outdir
    
    os.makedirs(outdir, exist_ok=True)
    set_logger(os.path.join(outdir, "rotation.log"))
    
    model = load_MouseCHD_model(conf_path=conf_path,
                                weights_path=os.path.join(os.path.dirname(conf_path), ckpt))
    target_size = model.layers[0].output_shape[0][1:]
    
    label_df = pd.read_csv(args.label)
        
    for theta_x in range(5, 360, 5):
        savename = f"rotation_{theta_x:02d}.csv"
        logging.info(f"theta_x: {theta_x}")
        
        df = predict_folder(model=model,
                            imdir=args.imdir,
                            maskdir=args.maskdir,
                            target_size=target_size,
                            label_df=label_df,
                            stage="eval",
                            batch_size=args.batch_size,
                            save=os.path.join(outdir, savename),
                            grouped_result=bool(args.grouped_result),
                            theta_x=theta_x)
        
        # Save summary results
        sum_df = summarize_results(df=df,
                                   ckpt=ckpt,
                                   test_fn=savename,
                                   sum_df=sum_df,
                                   grouped_result=args.grouped_result)
        
        sum_df.to_csv(sum_df_path, index=False)
        

parser = argparse.ArgumentParser()
parser.add_argument("-model_dir", help="model directory", default=None)
parser.add_argument("-fold", help="fold", choices=["F1", "F2", "F3", "F4", "F5"], default=None)
parser.add_argument("-imdir", help="image directory")
parser.add_argument("-maskdir", help="mask directory", default=None)
parser.add_argument("-label", help="path to label csv file", default=None)
parser.add_argument("-batch_size", type=int, help="batch size", default=8)
parser.add_argument("-outdir", help="output directory", default=None)
parser.add_argument("-grouped_result", help="grouping result?", type=int, choices=[0,1], default=0)
parser.add_argument("-device", help="device: 'cpu' or 'gpu'", choices=['cpu', 'gpu'], default='gpu')


args = parser.parse_args()
main(args)


""" 
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
F="F5"
singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/test_rotation.py \
    -model_dir "$WORKDIR/OUTPUTS/RotationInvariance" \
    -fold "$F" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -label "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -batch_size 8
    
singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/test_rotation.py \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -imdir "$WORKDIR/DATA/CTs/resampled/followup/images" \
    -label "$WORKDIR/DATA/CTs/labels/followup.csv" \
    -outdir "$WORKDIR/OUTPUTS/Classifier/all-in/results/rotation_followup" \
    -batch_size 8
    
singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/test_rotation.py \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -label "$WORKDIR/DATA/CTs/labels/Amaia.csv" \
    -outdir "$WORKDIR/OUTPUTS/Classifier/all-in/results/rotation_Amaia" \
    -batch_size 8
    
singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/test_rotation.py \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/retrain_refined" \
    -imdir "$WORKDIR/DATA/CTs/resampled" \
    -label "$WORKDIR/DATA/CTs/labels/divergence.csv" \
    -batch_size 8
"""
