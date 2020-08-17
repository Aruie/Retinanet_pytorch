
# from pycocotools.coco import COCO
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os


def create_coco_dataloader(coco_dir, batch_size, num_workers= 1) :

    train_data = CocoDetection(root = os.path.join(coco_dir, 'images','train2017'), 
                                annFile = os.path.join(root, 'annotations/instances_train2017.json') )

    val_data = CocoDetection(root = os.path.join(coco_dir, 'images','val2017'), 
                                annFile = os.path.join(coco_dir, 'annotations/instances_val2017.json') )

    train_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    return train_loader, val_loader



if __name__=="__main__" :
   
    coco_dir = 'd:/aru/datasets/coco/'
    create_coco_dataloader(coco_dir, 4, 1)
    print(val_data)