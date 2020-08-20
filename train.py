import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

from model.retinanet import RetinaNet
from model.losses import FocalLoss
from model.dataloader import CocoDataset, Normalizer, Resizer, AspectRatioBasedSampler, collater

import argparse
from time import perf_counter
import os



def main() :
    # Get argparser
    args = make_args()


    
    # 
    # if args.train == True :
    #     train(args.dataset, args.data_path, args.batch, args.epochs)
    # else :
    #     raise ValueError('Not Support "Train = False" Yet')
    train(args.dataset, args.data_path, args.batch, args.epochs)

def make_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'only "coco"', default = 'coco')
    parser.add_argument('--data_path', help = 'Path Datasets, default = "../datasets/coco/"', default = "../datasets/coco/")
    parser.add_argument('-t', '--train', help = 'only True', action = 'store_true')
    parser.add_argument('--batch', type = int, help='Number of MiniBatch, Default = 2', default = 2)
    parser.add_argument('--epochs', type = int, help='Number of Epochs, Default = 1', default = 1)
    args = parser.parse_args()
    return args
    
def make_dataloader(coco_dir, set_name, batch_size = 2, num_workers = 2) :

    if set_name == 'val2017' :
        dataset = CocoDataset(root_dir = coco_dir, set_name = set_name, transform = transforms.Compose([Normalizer(), Resizer()]))
    else :
        raise ValueError('TrainSet Service Not Yet')

    sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler)

    return dataloader


def train(dataset, data_path, batch_size = 2, epochs = 1) :


    # Make Dataloader
    if dataset == 'coco' :
        if 'images' not in os.listdir(data_path) :
            raise ValueError('"images" folder Not in Path')
    
        #dataloader_train = make_dataloader(coco_dir, 'train2017', batch_size = batch_size)    
        dataloader_val = make_dataloader(data_path, 'val2017', batch_size = batch_size)
        
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Make Model & Loss & optimizer
    model = RetinaNet().to(device)
    criterion = FocalLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay=0.0001, momentum=0.9)

    for epoch in range(1, epochs + 1) :

        print(f'Epoch : {epoch:10}')

        losses = []

        for step, data in enumerate(dataloader_val) : 

            t00 = perf_counter()
            
            image = data['img'].to(device)
            annotation = data['annot'].to(device)

            t0 = perf_counter()
            print(f'data to device : {t0-t00}')

            classification, regression, anchors = model(image)

            t1 = perf_counter()
            print(f'model : {t1-t0}')

            loss_classification, loss_regression = criterion(classification, regression, anchors, annotation)
            loss = loss_classification + loss_regression

            t2 = perf_counter()
            print(f'criterion : {t2-t1}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t3 = perf_counter()
            print(f'backward : {t3-t2}\n')


            if step % 100 == 1 :
                print(f'\tStep : {step+1:3}', end='\t')
                print(f'Class : {loss_classification},  Box : {loss_regression}')

            losses.append(loss.item())

        print(f'Epochs {epoch:6}\t Train Loss : {sum(losses) / len(losses) : .4f}\t Val Loss : Not Yet ')

    torch.save(model, f'Ep{epochs}.pkl')



if __name__ == '__main__' :

    main()
