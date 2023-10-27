#########################
# AUGMENTATION POLICIES #
#########################
from volumentations import (Compose,
                            Rotate,
                            GaussianNoise)

AUGMENT_POLS = [None, "augment0"]

def augment0(im, mask=None):
    """
    Augmentation for training data. (policy 0)
    Args:
        im (np.array): image array
    Returns:
        np.array: augmented image array
    """
    aug = Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.5),
    ], p=1.0)
    
    if mask is None:
        return aug(**{'image': im})['image']
    else:
        transformed = aug(**{'image': im, 'mask': mask})

        return transformed