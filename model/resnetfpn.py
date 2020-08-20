import torch
from torch import nn
import numpy as np

from torch.functional import F




def ResNet50_FPN(fpn_channel = 256) :
    return ResNetFPN([3,4,6,3], fpn_channel, bottlenect = True)

class ResNetFPN(nn.Module) : 
    def __init__(self, repeat_list, fpn_channel, bottlenect) :
        super().__init__()
        
        self.conv_intro = nn.Conv2d(3, 64, (7,7), stride = 2, padding = 3)
        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1 )
        self.bottleneck = bottlenect

        self.block1 = RepeatedBlock(self.bottleneck, 64, repeat_list[0], True)
        self.block2 = RepeatedBlock(self.bottleneck, 128, repeat_list[1])
        self.block3 = RepeatedBlock(self.bottleneck, 256, repeat_list[2])
        self.block4 = RepeatedBlock(self.bottleneck, 512, repeat_list[3])

        self.conv6 = nn.Conv2d(2048, fpn_channel, (3,3), stride = 2, padding = 1)
        self.conv7 = nn.Conv2d(fpn_channel, fpn_channel, (3,3), stride = 2, padding = 1)

        self.conv5 = nn.Conv2d(2048, fpn_channel, (1,1))
                
        self.conv4_up = nn.Conv2d(fpn_channel, fpn_channel, (3,3), padding = 1)
        self.conv4 = nn.Conv2d(1024, fpn_channel, (1,1))
        
        self.conv3_up = nn.Conv2d(fpn_channel, fpn_channel, (3,3), padding = 1)
        self.conv3 = nn.Conv2d(512, fpn_channel, (1,1))

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
    def __init__(self,  bottleneck, channel, repeat, is_first = False) :
        super(RepeatedBlock, self).__init__()
        
        self.chaanel = channel
        self.is_first = is_first
        self.repeat = repeat
        self.bottleneck = bottleneck

        self.blocks = nn.ModuleList()

        for i in range(self.repeat) :
            is_start = (True if i == 0 else False)
                

            # 수정필요  (ResidualBlockS구현안됨)  
            self.blocks.append( ResidualBlock(channel, is_start, is_first ))
            # if bottleneck == True :
            #     self.blocks.append( ResidualBlock(channel, is_start, is_first ))
            # else :
            #     self.blocks.append( ResidualBlockS(channel, is_start, is_first ))
            
    def forward(self, x) :
        for layer in self.blocks :
            x = layer(x)
        return x

class ResidualBlock(nn.Module) :
    def __init__(self, channel, is_start, is_first = False) :
        super(ResidualBlock, self).__init__()
        
        self.channel = channel
        self.is_start = is_start
        self.is_first = is_first

        if self.is_start == True : 
            if self.is_first == True : 
                self.conv1 = nn.Conv2d(self.channel, self.channel, (1,1))
                self.convsc = nn.Conv2d(self.channel, self.channel * 4, (1,1))
                self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), padding = 1)

            else : 
                self.conv1 = nn.Conv2d(self.channel*2, self.channel, (1,1))
                self.convsc = nn.Conv2d(self.channel*2, self.channel * 4, (1,1), stride=2)
                self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), stride=2, padding = 1)        
            

        else :
            self.conv1 = nn.Conv2d(self.channel*4, self.channel, (1,1))
            self.conv2 = nn.Conv2d(self.channel, self.channel, (3,3), padding = 1)       
            
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.act1 = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(self.channel)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.channel, self.channel * 4 , (1,1))
        self.bn3 = nn.BatchNorm2d(self.channel * 4)
        self.act3 = nn.ReLU()

    def forward(self, x) :
        sc = x
        if self.is_start == True :
            sc = self.convsc(sc)

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
    a = ResNet50_FPN()
    test_in = torch.randn(1,3,512,512)
    y = a(test_in)

    print(sum(p.numel() for p in a.parameters() if p.requires_grad))


    for i in y :
        print(i.shape)

        
