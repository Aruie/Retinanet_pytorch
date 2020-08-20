import torch
from torch import nn
import numpy as np

from model.resnetfpn import ResNet50_FPN
from model.anchor import Anchors
from model.losses import FocalLoss


class CustomConv2d(nn.Module) :
    def __init__(self, in_channel, out_channel, bias = 0, act = True) :
        super().__init__()
        
        self.layer = nn.Conv2d( in_channel, out_channel, (3,3), padding=1 )
        nn.init.normal_(self.layer.weight, std = 0.01)
        nn.init.constant_(self.layer.bias, bias)
        
        if act == True :
            self.act = nn.ReLU(inplace = True)
        
    def forward(self, x) :
        x = self.layer(x)
        if 'self.act' in locals() :
            x = self.act(x)
        return x
    

class BoxRegressor(nn.Module) :
    def __init__(self, channels = 256, num_anchor = 9) :
        super().__init__()

        self.anchor = num_anchor

        self.layers = nn.ModuleList([
            CustomConv2d(channels,channels), 
            CustomConv2d(channels,channels), 
            CustomConv2d(channels,channels), 
            CustomConv2d(channels,channels), 
            CustomConv2d(channels, self.anchor * 4, act = False) 
        ])

    def forward(self, x) :
        for layer in self.layers :
            x = layer(x)
        
        batch, _, height, width = x.shape
        # (B, C, H, W) to (B, H, W, anchor, 4)
        x = x.view(batch, self.anchor, 4, height, width)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(batch, -1, 4) 
        
        return x



class ObjectClassifier(nn.Module) : 
    def __init__(self, channels = 256, num_anchor = 9, num_classes = 80, prior = 0.01) :
        super().__init__()

        self.anchor = num_anchor
        self.classes = num_classes

        self.layers = nn.ModuleList([
                CustomConv2d(channels,channels), 
                CustomConv2d(channels,channels), 
                CustomConv2d(channels,channels), 
                CustomConv2d(channels,channels), 
                CustomConv2d(channels, self.anchor * self.classes, 
                    bias = -np.log((1 - prior) / prior), act = False )
            ])

    def forward(self, x) :
        for layer in self.layers :
            x = layer(x)

        batch, _, height, width = x.shape
        # (B, C, H, W) to (B, H, W, anchor, class)
        x = x.view(batch, self.anchor, self.classes, height, width)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(batch, -1, self.classes) 
        
        return x





class RetinaNet(nn.Module) :
    def __init__(self, channels = 256, num_anchor = 9, num_classes = 80) : 
        super().__init__()

        self.resnetfpn = ResNet50_FPN()
        self.classifier = ObjectClassifier(channels = channels, num_anchor = num_anchor, 
                                            num_classes = num_classes, prior=0.01)
        self.regressor = BoxRegressor(channels = channels, num_anchor = num_anchor)

        # 수정필요
        self.anchor = Anchors()

    def forward(self, image, annotations = None) :
        # if self.training and annotations is None :
        #     raise ValueError("In training mode, annotations must be requiments")

        fpn_out = self.resnetfpn(image)

        regression_out = torch.cat([self.regressor(x) for x in fpn_out], dim = 1)
        classification_out = torch.cat([self.classifier(x) for x in fpn_out], dim = 1)

        anchors = self.anchor(image)

        return classification_out, regression_out, anchors



if __name__ == '__main__' :
    # from anchor import Anchors
    # from resnetfpn import ResNet50_FPN

    model = RetinaNet()
    criterion = FocalLoss()

    test_input = torch.randn(1, 3, 512, 512)
    annotations = torch.randn(1, 5)

    classification, regression, anchors = model(test_input)
    loss = criterion(classification, regression, anchors, annotations)
    

    print(loss)

    
