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

    print(set_name, ' Images :', len(os.listdir(os.path.join(coco_dir,'images', set_name))))
    
    if set_name == 'val2017' :
        dataset = CocoDataset(root_dir = coco_dir, set_name = set_name, transform = transforms.Compose([Normalizer(), Resizer()]))
    else :
        raise ValueError('TrainSet Service Not Yet')

    sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler)

    return dataloader


def train(coco_dir, epochs = 1, batch_size = 2) :

    # from resnetfpn import ResNet50_FPN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = RetinaNet().to(device)
    criterion = FocalLoss().to(device)

    dataloader_val = make_dataloader(coco_dir, 'val2017', batch_size = batch_size)
    optimizer = optim.SGD(model.parameters(), lr = 0.1, weight_decay=0.0001, momentum=0.9)
    
    for epoch in range(1, epochs + 1) :

        print(f'Epoch : {epoch:10}')

        for step, data in enumerate(dataloader_val) : 

            print(f'Step : {step+1:10}', end='')

            image = data['img'].to(device)
            annotation = data['annot'].to(device)

            classification, regression, anchors = model(image)
            loss_classification, loss_regression = criterion(classification, regression, anchors, annotation)

            print(f'Class : {loss_classification},  Box : {loss_regression}')

            loss = loss_classification + loss_regression

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
         

if __name__ == '__main__' :

    train('../Datasets/coco/', batch_size = 1)