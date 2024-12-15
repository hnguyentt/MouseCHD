import os
import json
import re
import time
import pandas as pd
import logging
import argparse
import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from volumentations import (Compose,
                            Rotate,
                            GaussianNoise,
                            RandomRotate90,
                            RandomBrightnessContrast)

from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping,
                                        TensorBoard,
                                        CSVLogger)

from sklearn.utils import class_weight

from mousechd.classifier.utils import (MODEL_NAMES,
                                       CLF_DIR,
                                       find_best_ckpt,
                                       download_clf_models,
                                       calculate_metrics,
                                       load_label)
from mousechd.utils.tools import set_logger

from mousechd.datasets.utils import (resample3d, 
                              get_largest_connectivity,
                              crop_heart_bbx,
                              maskout_non_heart,
                              norm_min_max)
from mousechd.classifier.evaluate import predict_folder, summarize_results
from mousechd.classifier.models import MouseCHD


def augment1(im, mask=None):
    aug = Compose([
        Rotate((-180, 180), (0, 0), (0, 0), p=1),
        # RandomRotate90((1, 2), p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2,
                                 contrast_limit=0.2,
                                 brightness_by_max=True,
                                 p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.5),
    ], p=1.0)
    
    if mask is None:
        return aug(**{'image': im})['image']
    else:
        transformed = aug(**{'image': im, 'mask': mask})

        return transformed


class MouseCHDGen(tf.keras.utils.Sequence):
    def __init__(self,
                 imdir,
                 filenames,
                 batch_size,
                 target_size,
                 labels=None,
                 seed=42,
                 n_classes=1,
                 stage="train",
                 augment=None,
                 class_weights=None,
                 **kwargs):
        """Initialization

        Args:
            imdir (str): path to image directory
            filenames (ls): list of image files
            labels (ls): corresponding labels
            batch_size (int): batch size
            target_size (tuple): target size
            seed (int, optional): random seed. Defaults to 42.
            n_classes (int, optional): number of classes. Defaults to 1.
            stage (str, optional): 'train', 'test', 'val', or 'eval'. Defaults to "train".
            class_weights (tuple, optional): class weights. Defaults to None.
        """
        
        self.imdir = imdir
        self.filenames = filenames
        self.n_classes = n_classes
        if labels is None:
            labels = [0] * len(filenames)
            
        self.numeric_labels = labels
        if n_classes == 1:
            self.labels = np.array(labels)
        else:
            self.labels = tf.keras.utils.to_categorical(np.array(labels), num_classes=n_classes)
            
        self.list_IDs = [i for i in range(len(filenames))]
        self.batch_size = batch_size
        self.target_size = target_size
        self.seed = seed
        self.stage = stage
        self.augment = augment
        self.class_weights = class_weights
        self.kwargs = kwargs
        self.on_epoch_end()
        
    
    def __len__(self):
        return int(np.ceil(len(self.filenames)/float(self.batch_size)))
    

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        return self.__datagen(list_ids_tmp)
        
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.stage in ["train", "val"]:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
            
    
    def __datagen(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = np.empty((self.batch_size, self.n_classes))
        
        if self.class_weights is not None:
            weights = np.empty((self.batch_size, 1))
            
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            img = sitk.ReadImage(os.path.join(self.imdir, self.filenames[ID]))
            img = resample3d(img, self.target_size[:3][::-1])
            im = sitk.GetArrayFromImage(img)
            im = norm_min_max(im)
            
            if self.augment is not None:
                augment1(im, mask=(im != 0))
                
            if self.target_size[3] > 1:
                im = np.stack([im]*self.target_size[3], axis=3)
            else:
                im = np.expand_dims(im, axis=3)
                
            X[i,:] = im
            y[i,:] = self.labels[ID]
            
            files.append(self.filenames[ID])
            
            if self.class_weights is not None:
                weights[i,:] = self.class_weights[self.numeric_labels[ID]]
                
        
        if self.stage in ["test", "eval"]:
            return X, y, files
        else:
            if self.class_weights is None:
                return X, y
            else:
                return X, y, weights


def train_clf(save_dir,
              exp,
              data_dir, 
              label_dir, 
              configs,
              log_dir):     
    train_df = load_label(os.path.join(label_dir, "train.csv"), configs["seed"])
    val_df = load_label(os.path.join(label_dir,"val.csv"), configs["seed"])
    
    logging.info("TRAIN:")
    logging.info(train_df["label"].value_counts())
    logging.info("VAL:")
    logging.info(val_df["label"].value_counts())
    
    
    # optimizer
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=configs["lr"],
                                                                  decay_steps=1000,
                                                                  decay_rate=configs["decay"])
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_scheduler, 
                                        momentum = configs["momentum"], 
                                        nesterov = True)
    
    # loss_fn
    if configs["loss_fn"] == "categorical_crossentropy":
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # elif configs["loss_fn"] == "sigmoid_focal_crossentropy":
    #     loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=configs["alpha"], gamma=configs["gamma"])
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
    # class weights
    if configs["class_weights"]:
        weights = class_weight.compute_class_weight(class_weight="balanced", 
                                                    classes=np.unique(train_df["label"]), 
                                                    y=train_df["label"].values)
        weights = {i: weights[i] for i in range(len(train_df["label"].unique()))}
    else:
        weights = None
        
    logging.info("Class weights: {}".format(weights))
    
    # model
    model = MouseCHD(model_name=configs["model_name"],
                     input_size=configs["input_size"],
                     n_classes=configs["n_classes"],
                     first_filters=configs["first_filters"],
                     mask_depth=configs["mask_depth"],
                     is_bn_mask=configs["is_bn_mask"]).build_model()
        
    # Metrics
    metrics = ["accuracy", 
               tf.keras.metrics.Recall(), 
               tf.keras.metrics.Precision()]
    
    # Callbacks
    if configs["save_best"]:
        save_model_name = "best_model.hdf5"
    else:
        save_model_name = "epoch-{epoch:03d}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_dir, save_model_name),
                                 monitor=configs["monitor"],
                                 verbose=1,
                                 save_best_only=configs["save_best"],
                                 save_weights_only=True,
                                 save_freq="epoch")
    early_stop = EarlyStopping(configs["monitor"], patience=configs["patience"])
    tb = TensorBoard(log_dir=os.path.join(log_dir, exp),
                     update_freq="batch")
    csv_logger = CSVLogger(os.path.join(save_dir, "train.csv"), append=True)
    
    # Data generators
    train_gen = MouseCHDGen(imdir=data_dir,
                            filenames=train_df["heart_name"].values,
                            batch_size=configs["batch_size"],
                            target_size=configs["input_size"],
                            labels=train_df["label"].values,
                            seed=configs["seed"],
                            n_classes=configs["n_classes"],
                            stage="train",
                            augment=configs["augment"],
                            class_weights=weights)
    val_gen = MouseCHDGen(imdir=data_dir,
                          filenames=val_df["heart_name"].values,
                          batch_size=configs["batch_size"],
                          labels=val_df["label"].values,
                          seed=configs["seed"],
                          target_size=configs["input_size"],
                          n_classes=configs["n_classes"],
                          stage="val",
                          augment=configs["augment"],
                          class_weights=weights) 
    
    # Evaluate initial model and resume   
    if configs["resume"] is not None:
        try:
            model.load_weights(os.path.join(save_dir, configs["resume"]))
        except FileNotFoundError:
            logging.info("Resumed weights not found, retrain from default weights")
            download_clf_models()
            model.load_weights(os.path.join(CLF_DIR, "best_model.hdf5"))
            
        try:
            initial_epoch = int(configs["resume"].split("-")[1])
        except IndexError:
            initial_epoch = 0
    else:
        initial_epoch = 0
        model.save_weights(os.path.join(save_dir, "initial_weights.hdf5"))
        if not os.path.isfile(os.path.join(save_dir, "train.csv")):
            logging.info("="*15 + "//" + "="*15)
            logging.info("Evaluate initial model:")
            logging.info("On training: ")
            train_res = predict_folder(model=model,
                                       imdir=data_dir,
                                       maskdir=None,
                                       target_size=configs["input_size"],
                                       label_df=train_df,
                                       stage="eval",
                                       batch_size=configs["batch_size"],
                                       save=None,
                                       grouped_result=False)
            val_res = predict_folder(model=model,
                                     imdir=data_dir,
                                     maskdir=None,
                                     target_size=configs["input_size"],
                                     label_df=val_df,
                                     stage="eval",
                                     batch_size=configs["batch_size"],
                                     save=None,
                                     grouped_result=False)
            df =  pd.DataFrame(columns=["epoch", 
                                        "loss", 
                                        "accuracy", 
                                        "recall", 
                                        "precision", 
                                        "weighted_accuracy",
                                        "val_loss", 
                                        "val_accuracy", 
                                        "val_recall", 
                                        "val_precision", 
                                        "val_weighted_accuracy"])
            train_loss = tf.keras.metrics.binary_crossentropy(tf.constant(train_res["label"].values.astype(float)),
                                                              tf.constant(train_res["prob"].values.astype(float)))
            val_loss = tf.keras.metrics.binary_crossentropy(tf.constant(val_res["label"].values.astype(float)),
                                                            tf.constant(val_res["prob"].values.astype(float)))
            train_metrics = calculate_metrics(train_res["prob"].values.astype(float),
                                              train_res["label"].values.astype(float))
            val_metrics = calculate_metrics(val_res["prob"].values.astype(float),
                                            val_res["label"].values.astype(float))
            df.loc[len(df), :] = [0, 
                                  train_loss.numpy(), train_metrics["acc"], train_metrics["sens"], train_metrics["spec"], train_metrics["bal_acc"],
                                  val_loss.numpy(), val_metrics["acc"], val_metrics["sens"], val_metrics["spec"], val_metrics["bal_acc"]]
            
            df.to_csv(os.path.join(save_dir, "train.csv"), index=0)
    
    if weights is not None:
        model.compile(loss=loss_fn, 
                      optimizer=optimizer,
                      metrics=metrics,
                      weighted_metrics=["accuracy"])
    else:
        model.compile(loss=loss_fn,
                      optimizer=optimizer,
                      metrics=metrics)
        
    # Training    
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=configs["epochs"],
                        verbose=1,
                        callbacks=[checkpoint, early_stop, tb, csv_logger],
                        initial_epoch=initial_epoch)
    
    return model    


parser = argparse.ArgumentParser()
parser.add_argument("-exp_dir", type=str, help="experiment directory")
parser.add_argument("-exp", type=str, help="name of experiment")
parser.add_argument("-data_dir", type=str, help="path to data directory")
parser.add_argument("-label_dir", type=str, help="path to directory containing label files")
parser.add_argument("-test_path", type=str, help="path to test file", default=None)
parser.add_argument("-testdir", type=str, help="test datadir", default=None)
parser.add_argument("-test_bz", type=int, help="test batch size", default=16)
parser.add_argument("-configs", type=str, help="path to configs file", default=None)
parser.add_argument("-log_dir", type=str, help="Logging directory for tensorboard", default=None)
parser.add_argument("-evaluate", help="evaluate: best, none, all", type=str,
                    choices=["best", "none", "all"], default="none")
parser.add_argument("-logfile", type=str, help="path to logfile", default=None)
parser.add_argument("-epochs", type=int, help="epochs", default=None)
    
MONITOR_LIST = ["val_loss","val_accuracy", "val_weighted_accuracy"]
AUGMENT_POLS = [None, "augment0", "augment1"]

def main(args):
    
    download_clf_models()
    # Process arguments
    if args.configs is None:
        configs = json.load(open(os.path.join(CLF_DIR, "configs.json"), "r"))
    else:
        configs = json.load(open(args.configs, "r"))
    if args.epochs is not None:
        configs["epochs"] = args.epochs
        
    if configs["n_classes"] == 1:
        assert configs["loss_fn"] in ["binary_crossentropy","sigmoid_focal_crossentropy"], "loss_fn for n_classes=1: binary_crossentropy,sigmoid_focal_crossentropy"
    else:
        assert configs["loss_fn"] == "categorical_crossentropy", "n_classes={}, only categorical_crossentropy can be applied as loss_fn".format(configs["n_classes"])
    
    assert configs["model_name"] in MODEL_NAMES, f"'model_name' must be in {MODEL_NAMES}"
    assert configs["mask_depth"] in [i for i in range(1,5)], "'mask_depth' must be in {}".format([i for i in range(1,5)])
    assert configs["augment"] in AUGMENT_POLS, f"'augment' must be in {AUGMENT_POLS}"
    assert configs["monitor"] in MONITOR_LIST, f"'monitor' must be in {MONITOR_LIST}"
    
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = os.path.join(args.label_dir, "test.csv")
     
    
    save_dir = os.path.join(args.exp_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if args.logfile is not None:
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        set_logger(args.logfile)
    else:
        set_logger(os.path.join(save_dir, "training.log"))
    
    with open(os.path.join(save_dir, "configs.json"),"w") as f:
        json.dump(configs, f, indent=1)
    
    # TRain  
    strat_time = time.time()
    logging.info("="*15 + "//" + "="*15)
    logging.info("TRAIN")
    if args.log_dir is None:
        log_dir = os.path.join(save_dir, "LOGS")
    else:
        log_dir = args.log_dir 
    model = train_clf(save_dir=save_dir,
                      exp=args.exp,
                      data_dir=args.data_dir,
                      label_dir=args.label_dir, 
                      configs=configs,
                      log_dir=log_dir)
    end_time = time.time()
    logging.info("Training time (hours): {}".format((end_time - strat_time) / 3600))
    
    # Evaluate
    if args.evaluate != "none":
        logging.info("="*15 + "//" + "="*15)
        if args.testdir is None:
            testdir = args.data_dir
        else:
            testdir = args.testdir
        logging.info("EVALUATE")
        sum_df_path = os.path.join(save_dir, "results", "summary.csv")
        try:
            sum_df = pd.read_csv(sum_df_path)
        except FileNotFoundError:
            sum_df = pd.DataFrame(columns=["ckpt", "testfile",
                                        "acc", "bal_acc",
                                        "sens", "spec", "auc"])
        df = pd.read_csv(test_path)
        ## Restore ckpt
        if args.evaluate == "best":
            ckpts = [find_best_ckpt(save_dir, re.sub(r"^val_", "", configs["monitor"]))]
        else:
            ckpts = sorted([x for x in os.listdir(save_dir)
                            if x.endswith(".hdf5") and (not x.startswith("."))])
        
        for ckpt in ckpts:
            logging.info("Restore: {}".format(ckpt))
            model.load_weights(os.path.join(save_dir, ckpt))
            
            res = predict_folder(model=model,
                                imdir=testdir,
                                maskdir=None,
                                target_size=configs["input_size"],
                                label_df=df,
                                stage="eval",
                                batch_size=args.test_bz,
                                save=os.path.join(save_dir, "results", ckpt, os.path.basename(test_path)),
                                grouped_result=True)
            
            sum_df = summarize_results(df=res,
                                    ckpt=ckpt,
                                    test_fn=os.path.basename(test_path),
                                    sum_df=sum_df,
                                    grouped_result=True)
            
            sum_df.to_csv(sum_df_path, index=False)
            
args = parser.parse_args()
main(args)
            
""" 
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
SRCDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD"
F="F5"
singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/train_RotInv.py \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "RotationInvariance/$F" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/Imagine/images_x5" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/5folds/$F" \
    -test_path "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -configs "$SRCDIR/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"
"""