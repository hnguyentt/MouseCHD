import os, sys
import shutil
from time import time
from pathlib import Path

from nnunet.inference.predict import predict_from_folder, predict_cases
from nnunet.paths import (default_plans_identifier, 
                          network_training_output_dir, 
                          default_cascade_trainer, 
                          default_trainer)
from batchgenerators.utilities.file_and_folder_operations import join, isdir, save_json, load_pickle
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


from ..utils.tools import (CACHE_DIR,
                           BASE_URI,
                           download_file)

SEG_DIR = os.path.join(CACHE_DIR, "HeartSeg")

urls = {
    "plans.pkl": f"{BASE_URI}/segmentation/plans.pkl",
    "fold_0/model_final_checkpoint.model": f"{BASE_URI}/segmentation/fold_0/model_final_checkpoint.model",
    "fold_0/model_final_checkpoint.model.pkl": f"{BASE_URI}/segmentation/fold_0/model_final_checkpoint.model.pkl",
    "fold_0/postprocessing.json": f"{BASE_URI}/segmentation/fold_0/postprocessing.json",
    "fold_1/model_final_checkpoint.model": f"{BASE_URI}/segmentation/fold_1/model_final_checkpoint.model",
    "fold_1/model_final_checkpoint.model.pkl": f"{BASE_URI}/segmentation/fold_1/model_final_checkpoint.model.pkl",
    "fold_1/postprocessing.json": f"{BASE_URI}/segmentation/fold_1/postprocessing.json",
    "fold_2/model_final_checkpoint.model": f"{BASE_URI}/segmentation/fold_2/model_final_checkpoint.model",
    "fold_2/model_final_checkpoint.model.pkl": f"{BASE_URI}/segmentation/fold_2/model_final_checkpoint.model.pkl",
    "fold_2/postprocessing.json": f"{BASE_URI}/segmentation/fold_2/postprocessing.json",
    "fold_3/model_final_checkpoint.model": f"{BASE_URI}/segmentation/fold_3/model_final_checkpoint.model",
    "fold_3/model_final_checkpoint.model.pkl": f"{BASE_URI}/segmentation/fold_3/model_final_checkpoint.model.pkl",
    "fold_3/postprocessing.json": f"{BASE_URI}/segmentation/fold_3/postprocessing.json",
    "fold_4/model_final_checkpoint.model": f"{BASE_URI}/segmentation/fold_4/model_final_checkpoint.model",
    "fold_4/model_final_checkpoint.model.pkl": f"{BASE_URI}/segmentation/fold_4/model_final_checkpoint.model.pkl",
    "fold_4/postprocessing.json": f"{BASE_URI}/segmentation/fold_4/postprocessing.json",
}

part_id = 0
num_parts = 1
save_npz = False
lowres_segmentations = None
num_threads_preprocessing = 6
num_threads_nifti_save = 2
disable_tta = True
step_size = 0.5
overwrite_existing = False
mode = "normal"
all_in_gpu = None
model = "3d_fullres"
trainer_class_name = default_trainer
cascade_trainer_class_name = default_cascade_trainer
task_name = 113
disable_mixed_precision = False
chk = "model_final_checkpoint"


def segment_from_folder(indir, outdir, folds=None):
    # By default --> doing the ensemble. Specify fold by folds=1
    if folds is not None:
        folds = [folds]
    for fname, url in urls.items():
        fname = Path(fname)
        download_file(url, fname, cache_dir=SEG_DIR)
        
    trainer = trainer_class_name
    model_folder_name = SEG_DIR
    
    st = time()
    predict_from_folder(model_folder_name, 
                        indir, 
                        outdir, 
                        folds, 
                        save_npz, 
                        num_threads_preprocessing,
                        num_threads_nifti_save, 
                        lowres_segmentations, 
                        part_id, 
                        num_parts, 
                        not disable_tta,
                        overwrite_existing=overwrite_existing, 
                        mode=mode, 
                        overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, 
                        checkpoint_name=chk)
    end = time()
    save_json(end - st, join(outdir, 'prediction_time.txt'))
    
     
    