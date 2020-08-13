import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

from model.retinanet import RetinaNet
from model.losses import FocalLoss
from model.dataloader import CocoDataset, Normalizer, Resizer, AspectRatioBasedSampler, collater


def make_dataloader(coco_dir, set_name, batch_size = 2, num_workers = 2) :

    # coco_dir = '/content/coco/'
    # os.chdir('/content/Retinanet_pytorch/')

    print(set_name, ' Images :', len(os.listdir(os.path.join(cocodir, set_name))))
    
    if set_name == 'val2017' :
        dataset = CocoDataset(root_dir = coco_dir, set_name = set_name, transform = transforms.Compose([Normalizer(), Resizer()]))
    else :
        raise ValueError('TrainSet Service Not Yet')

    sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler)

    return dataloader


def train(coco_dir, epochs = 1) :

    # from resnetfpn import ResNet50_FPN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = RetinaNet().to(device)
    criterion = FocalLoss().to(device)

    make_dataloader(coco_dir, 'val2017')
    # test_input = torch.randn(1, 3, 512, 512)
    # annotations = torch.randn(1, 5)

    for epoch in range(1, epochs + 1) :

        for step, data in enumerate(dataloader_val) : 

            image = data['img'].to(device)
            annotation = data['annot'].to(device)

            regression, classification, anchors = model(image)
            loss = criterion(regression, classification, anchors, annotation)



            print(loss.shape)
            if step > 1 :
                break


if __name__ == '__main__' :

    train('./coco/')