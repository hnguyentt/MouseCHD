
""" 
Evaluate models
"""
import logging
import os, re
import pandas as pd
import numpy as np
import SimpleITK as sitk

from .datagens import MouseCHDEvalGen
from .utils import calculate_metrics, eval_clf
from ..datasets.utils import resample3d, norm_min_max
from .gradcam import GradCAM3D
from .models import load_MouseCHD_model


#TODO: Delete later
def predict_heart(conf_path, weights_path, resampled_im):
    model = load_MouseCHD_model(conf_path=conf_path,
                                weights_path=weights_path)
    
    input_shape = model.layers[0].output_shape[0][1:4]
    img = sitk.GetImageFromArray(resampled_im)
    img.SetSpacing((0.02, 0.02, 0.02))
    img = resample3d(img, input_shape[::-1])
    im = sitk.GetArrayFromImage(img)
    im = norm_min_max(im)
    im = np.expand_dims(im, axis=3)
    im = np.expand_dims(im, axis=0)
    
    preds = model.predict(tf.convert_to_tensor(im))[0]
    
    # GradCAM
    class_idx = np.argmax(preds)
    grad_model = GradCAM3D(model)
    gradcam = grad_model.compute_heatmap(im, classIdx=class_idx, upsample_size=resampled_im.shape)
    
    return preds, gradcam


def predict_folder(model,
                   imdir,
                   maskdir,
                   target_size,
                   label_df=None,
                   stage="test",
                   batch_size=8,
                   save=None,
                   grouped_result=False):
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
                               stage=stage)
    
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
                                   stage=stage)
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


def summarize_results(df, ckpt, test_fn, sum_df, grouped_result=False):
    """Summarize evaluation results

    Args:
        df (pd.DataFrame): evaluation result
        ckpt (str): name of checkpoint
        test_fn (str): test filename
        sum_df (pd.DataFrame): summarized dataframe

    Returns:
        pd.DataFrame: updated summarized dataframe
    """
    if grouped_result:
        df["heart_name"] = df["heart_name"].str.replace(r"_\d+$", "")
        group_df = df.groupby(by="heart_name").agg("mean")
        group_df["pred"] = (group_df["prob"] > 0.5)*1
        df = group_df
    
    res_metrics = calculate_metrics(df["prob"].values.astype(float),
                                    df["label"].values.astype(float))
    
    sum_df.loc[len(sum_df), :] = [ckpt,
                                  test_fn,
                                  res_metrics["acc"],
                                  res_metrics["bal_acc"],
                                  res_metrics["sens"],
                                  res_metrics["spec"],
                                  res_metrics["auc"]]
        
    return sum_df
    