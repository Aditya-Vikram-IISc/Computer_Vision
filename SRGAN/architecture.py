import math
import torch
import torch.nn as nn


__all_modules__ = ["Discriminator_Block", "Generator_Residual_Block", "Generator_Upsample_Block"]




#1: Discriminator Building Blocks

class Discriminator_Block(nn.Module):
    '''
    Note: Building block of SRGAN's discriminator
    '''
    def __init__(self, in_channels, out_channels, stride):
        super(Discriminator_Block, self).__init__()

        self.block = nn.Sequential(
                                nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3, stride= stride, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(negative_slope= 0.2)       #Value suggested in the paper
                                    )
        
    def forward(self, x:torch.Tensor):
        out = self.block(x)
        return out


#2: Generator Building Blocks

class Generator_Residual_Block(nn.Module):
    def __init__(self):
        super(Generator_Residual_Block, self).__init__()

        self.block = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 1),
                            nn.BatchNorm2d(64),
                            nn.PReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 1),
                            nn.BatchNorm2d(64)
                                    )
        
    def forward(self, x:torch.Tensor):

        y = self.block(x)
        out = torch.add(x,y)
        return out


class Generator_Upsample_Block(nn.Module):
    def __init__(self, upscale_factor:int = 2):
        super(Generator_Upsample_Block, self).__init__()

        self.block = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
                            nn.PixelShuffle(upscale_factor),
                            nn.PReLU()
                                    )
        
    def forward(self, x:torch.Tensor):

        out = self.block(x)
        return out