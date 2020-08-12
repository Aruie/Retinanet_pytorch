import torch

from model.retinanet import RetinaNet
from model.dataloader import CocoDataset


def make_dataloader(root, set_name) :
    dataset_val = CocoDataset(root, set_name, transform = transforms.Compose([Normalizer(), Resizer()]))



def train





if __name__ == '__main__' :

    