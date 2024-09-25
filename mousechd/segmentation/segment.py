from time import time
from pathlib import Path


def segment_from_folder(indir, 
                        outdir, 
                        folds=None, 
                        step_size=0.5, 
                        disable_mixed_precision=False,
                        num_threads_preprocessing=6,
                        num_threads_nifti_save=2):
    """Segment mouse heart with nnUNet

    Args:
        indir (str): input directory
        outdir (str): output directory
        folds (int, optional): fold number to segment, otherwise it's ensembling 5 folds. Defaults to None.
    """
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import (default_cascade_trainer, 
                            default_trainer)
    from batchgenerators.utilities.file_and_folder_operations import join, save_json

    from .utils import SEG_DIR, download_seg_models

    part_id = 0
    num_parts = 1
    save_npz = False
    lowres_segmentations = None
    num_threads_preprocessing = num_threads_preprocessing
    num_threads_nifti_save = num_threads_nifti_save
    disable_tta = True
    step_size = step_size
    overwrite_existing = False
    mode = "normal"
    all_in_gpu = None
    model = "3d_fullres"
    trainer_class_name = default_trainer
    cascade_trainer_class_name = default_cascade_trainer
    task_name = 113
    disable_mixed_precision = disable_mixed_precision
    chk = "model_final_checkpoint"
    
    
    # By default --> doing the ensemble. Specify fold by folds=1
    if folds is not None:
        folds = [folds]
    download_seg_models()
        
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
    
     
    