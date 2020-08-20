import torch
from torch import nn
import numpy as np

from torch.functional import F




def ResNet50_FPN(fpn_channel = 256) :
    return ResNetFPN([3,4,6,3], fpn_channel, bottlenect = True)

def ResNet50_FPN_Mini(fpn_channel = 256) :
    return ResNetFPN([3,4,6,3], fpn_channel, bottlenect = True, const_channels = True)

class ResNetFPN(nn.Module) : 
    def __init__(self, repeat_list, fpn_channel, bottlenect, const_channels = False) :
        super().__init__()
        
        self.conv_intro = nn.Conv2d(3, 64, (7,7), stride = 2, padding = 3)
        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1 )
        self.bottleneck = bottlenect
        
        if const_channels :
            in_channels = [64,128,256,256]
            out_channels = [128,256,256,256]
        else : 
            in_channels = [64,256,512,1024]
                
            if self.bottleneck : 
                out_channels = [256,512,1024,2048]
            else :
                out_channels = in_channels

        self.block1 = RepeatedBlock(self.bottleneck, in_channels[0], out_channels[0], repeat_list[0], is_first=True)
        self.block2 = RepeatedBlock(self.bottleneck, in_channels[1], out_channels[1], repeat_list[1])
        self.block3 = RepeatedBlock(self.bottleneck, in_channels[2], out_channels[2], repeat_list[2])
        self.block4 = RepeatedBlock(self.bottleneck, in_channels[3], out_channels[3], repeat_list[3])

        self.conv6 = nn.Conv2d(out_channels[3], fpn_channel, (3,3), stride = 2, padding = 1)
        self.conv7 = nn.Conv2d(fpn_channel, fpn_channel, (3,3), stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(out_channels[3], fpn_channel, (1,1))
        self.conv4_up = nn.Conv2d(fpn_channel, fpn_channel, (3,3), padding = 1)
        self.conv4 = nn.Conv2d(out_channels[2], fpn_channel, (1,1))
        self.conv3_up = nn.Conv2d(fpn_channel, fpn_channel, (3,3), padding = 1)
        self.conv3 = nn.Conv2d(out_channels[1], fpn_channel, (1,1))
    

    def  forward(self, x) :

        x = self.conv_intro(x)
        x = self.pool(x)

        x = self.block1(x)
        out3 = self.block2(x)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        
        p5 = self.conv5(out5)
        p4 = self.conv4(out4) + self.conv4_up(F.interpolate(p5, scale_factor = 2))
        p3 = self.conv3(out3) + self.conv3_up(F.interpolate(p4, scale_factor = 2))
        
        p6 = self.conv6(out5)
        p7 = self.conv7(F.relu(p6))

        return p3, p4, p5, p6, p7

# Repeat Block
class RepeatedBlock(nn.Module) :
    def __init__(self,  bottleneck, in_channel, out_channel, repeat, is_first = False) :
        super(RepeatedBlock, self).__init__()
        
        self.is_first = is_first
        self.repeat = repeat
        self.bottleneck = bottleneck

        is_identity = True if is_first == False else False
        self.blocks = nn.ModuleList()

        # 첫번째 블록
        self.blocks.append( ResidualBlock(in_channel, out_channel, is_identity = is_identity, is_start = True))

        for i in range(self.repeat - 1) :
            self.blocks.append( ResidualBlock(out_channel, out_channel))
    
            # 수정필요  (ResidualBlockS구현안됨)  
                        
            # if bottleneck == True :
            #     self.blocks.append( ResidualBlock(channel, is_start, is_first ))
            # else :
            #     self.blocks.append( ResidualBlockS(channel, is_start, is_first ))
            
    def forward(self, x) :
        for layer in self.blocks :
            x = layer(x)
        return x

class ResidualBlock(nn.Module) :
    def __init__(self, in_channel, out_channel, is_identity = False, is_start = False) :
        super(ResidualBlock, self).__init__()
        
        self.is_identity = is_identity
        self.is_start = is_start
        mid_channel = int(out_channel / 4)

        if self.is_identity : 
            self.convsc = nn.Conv2d(in_channel, out_channel, (1,1), stride=2)
            self.conv1 = nn.Conv2d(in_channel, mid_channel, (1,1))
            self.conv2 = nn.Conv2d(mid_channel, mid_channel, (3,3), stride=2, padding = 1)        
            
        else : 
            if is_start == True :
                self.convsc = nn.Conv2d(in_channel, out_channel, (1,1))
            self.conv1 = nn.Conv2d(in_channel, mid_channel, (1,1))
            self.conv2 = nn.Conv2d(mid_channel, mid_channel, (3,3), padding = 1)
    
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(mid_channel, out_channel, (1,1))
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.act3 = nn.ReLU()


    def forward(self, x) :
        
        if self.is_identity | self.is_start :
            sc = self.convsc(x)
        else :
            sc = x
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = x + sc

        x = self.act3(x)
        return x


if __name__ == '__main__' :
    a = ResNet50_FPN_Mini()
    
    # a = ResNet50_FPN()
    test_in = torch.randn(1,3,224,224)
    y = a(test_in)

    print(sum(p.numel() for p in a.parameters() if p.requires_grad))

    for i in y :
        print(i.shape)

        
