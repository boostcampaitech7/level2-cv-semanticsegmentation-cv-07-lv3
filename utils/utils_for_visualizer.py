import numpy as np


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def gender_to_eng(kor):
    if '여' in kor:
        return 'female'
    elif '남' in kor:
        return 'male'
    else:
        return 'unknown'