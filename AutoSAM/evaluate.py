import glob
import pickle

import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
import argparse


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    # args.save_dir = 'output_experiment/sam_unet_seg_ACDC_f0_tr_75'
    # test_acdc(args)
