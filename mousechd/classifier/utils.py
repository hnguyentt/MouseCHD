import os, re
import numpy as np
import pandas as pd
from sklearn import metrics
import logging
from pathlib import Path

from ..utils.tools import BASE_URI, CACHE_DIR, download_file

CLF_DIR = os.path.join(CACHE_DIR, "Classifier")
MODEL_NAMES = ["simple3d", "roimask3d", "roimask3d1"]

clf_urls = {
    "best_model.hdf5": f"{BASE_URI}/clf/best_model.hdf5",
    "configs.json": f"{BASE_URI}/clf/configs.json"
}


def download_clf_models():
    """Download classifiers
    """
    for fname, url in clf_urls.items():
        fname = Path(fname)
        download_file(url, fname, cache_dir=CLF_DIR, update=True)
        

def calculate_metrics(outputs, targets) -> dict:
    results = {}
    if len(outputs.shape) == 1:
        n_classes = 1
    else:
        n_classes = outputs.shape[1]
        
    if n_classes == 1:
        preds = (outputs > 0.5)*1
    else:
        preds = outputs.argmax(axis=1)
      
    acc = metrics.accuracy_score(targets, preds)
    bal_acc = metrics.balanced_accuracy_score(targets, preds)
    
    recalls = metrics.recall_score(targets, preds, 
                                   average=None, 
                                   labels=range(n_classes),
                                   zero_division=0.)
    
    f1s = metrics.f1_score(targets, preds,
                           average=None,
                           labels=range(n_classes),
                           zero_division=0.)
    f1 = np.mean(f1s)
    
    if n_classes < 3:
        try:
            if n_classes == 1:
                fpr, tpr, _ = metrics.roc_curve(targets, outputs)
            else:
                fpr, tpr, _ = metrics.roc_curve(targets, outputs[:, 1])
            aucs = [metrics.auc(fpr, tpr)]
        except ValueError:
            aucs = [0]
        
        specificities = [metrics.recall_score((1-targets)*1, (1-preds)*1, zero_division=0.)]
        recalls = np.array([metrics.recall_score(targets, preds, zero_division=0.)])
        f1s = np.array([metrics.f1_score(targets, preds, zero_division=0.)])
    else:
        aucs = []
        specificities = []
        for c in range(n_classes):
            try:
                fpr, tpr, _ = metrics.roc_curve((targets==c)*1, outputs[:, c])
                aucs.append(metrics.auc(fpr, tpr))
            except ValueError:
                aucs.append(0)
            specificities.append(metrics.recall_score((targets!=c)*1, (preds!=c)*1, zero_division=0.))
            
        recalls = metrics.recall_score(targets, preds, 
                                       average=None, 
                                       labels=range(n_classes),
                                       zero_division=0.)
        f1s = metrics.f1_score(targets, preds,
                           average=None,
                           labels=range(n_classes),
                           zero_division=0.)
    
    aucs = np.array(aucs)
    specificities = np.array(specificities)
    f1 = np.mean(f1s)
    recall = np.mean(recalls)
    specificity = np.mean(specificities)
    
    results['acc'] = acc
    results['bal_acc'] = bal_acc
    results['all_sens'] = recalls
    results['all_spec'] = specificities
    results['all_f1'] = f1s
    results['all_auc'] = aucs
    results['sens'] = recall
    results['spec'] = specificity
    results['f1'] = f1
    results['auc'] = np.mean(aucs)
    
    return results


def eval_clf(model,
             datagen,
             batch_size,
             stage="test",
             df=None,
             save=None):
    if df is None:
        df = pd.DataFrame(columns=["heart_name",
                                   "label",
                                   "prob",
                                   "pred"])
    y_preds = []
    y_trues = []
        
    for i, (X, y, imnames) in enumerate(datagen):
        if batch_size != 1:
            logging.info("Predict: {} - {}".format(i*batch_size+1,
                                                (i+1)*batch_size))
        else:
            logging.info(f"Predict: {batch_size}")
        
        if model.layers[-1].output_shape[1] == 1:
            idx = 0
        else:
            idx = 1
        probs = model.predict(X)[:, idx]
        preds = (probs > 0.5).astype(int)
        
        y_preds += list(preds)
        y_trues += y
        
        if stage == "eval":
            eval_metrics = calculate_metrics(np.array(probs), np.array(y))
            logging.info(f"Eval metrics: {eval_metrics}")
        
        tmp_df = pd.DataFrame({
            "heart_name": list(imnames),
            "label": list(y),
            "prob": list(probs),
            "pred": list(preds)
        })
        df = pd.concat([df, tmp_df], ignore_index=True)
        
        if save is not None:
            df.to_csv(save, index=False)
            
    return df


def load_label(path, seed=42):
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1,random_state=seed).reset_index(drop=True)
    
    return df


def find_best_ckpt(savedir, monitor):
    monitor_idx = {"loss": 2, 
                   "accuracy": 3, 
                   "recall": 4, 
                   "precision": 5,
                   "weighted_accuracy": 3}
    
    all_ckpts = sorted([re.sub(r".hdf5$", "", x) for x in os.listdir(savedir) 
                        if x.endswith(".hdf5") and (not x.startswith("."))])
    all_ckpts = [x for x in all_ckpts if x != "best_model"]
    monitor_values = [float(x.split("-")[monitor_idx[monitor]]) for x in all_ckpts]
    
    sort_indices = np.argsort(monitor_values)
    sorted_ckpts = [all_ckpts[i] for i in sort_indices]
    if monitor == "loss":
        return sorted_ckpts[0] + ".hdf5"
    else:
        return sorted_ckpts[-1] + ".hdf5"
    