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


def decode_rle_to_mask(rle, height, width):
    """
    rle: run-length encoded string
    height, width: dimensions of the mask
    Returns decoded binary mask
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def gender_to_eng(kor):
    if '여' in kor:
        return 'female'
    elif '남' in kor:
        return 'male'
    else:
        return 'unknown'
    

def reformat_metadata(metadata_row, class_count=0, annotation_count=0):
    metadata_row = metadata_row.fillna("N/A")  # NaN 값을 "N/A"로 대체
    gender_kor = metadata_row['성별'].values[0]

    gender_eng = gender_to_eng(gender_kor)

    # 폰트에 한국어 지원이 안돼서 영어로 출력 필요
    metadata_text = (
        f"ID: {metadata_row['ID'].values[0]} \n"
        f"age: {metadata_row['나이'].values[0]}\n"
        f"gender: {gender_eng}\n"
        f"weight: {metadata_row['체중(몸무게)'].values[0]}\n"
        f"height: {metadata_row['키(신장)'].values[0]}\n"
        f"issue: {metadata_row['Unnamed: 5'].values[0]}\n"
    )

    if class_count > 0:
        metadata_text += f"class: {len(class_count)}/{annotation_count}"
    
    return metadata_text