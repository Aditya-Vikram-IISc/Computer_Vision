import math
import torch
import torch.nn as nn
from architecture import Discriminator_Block, Generator_Residual_Block, Generator_Upsample_Block		#Blocks from architecture.py 



__all_models__ = ["Generator", "Discriminator"]


#1: Generator

class Generator(nn.Module):

    def __init__(self, image_channels:int =3, residual_count:int = 16, upscale:int = 4):
        super(Generator, self).__init__()

        self.upsample_counts = int(math.log2(upscale))

        #Block1
        self.block1 = nn.Sequential(
                                nn.Conv2d(in_channels=image_channels, out_channels= 64, kernel_size=9, stride=1, padding=4),
                                nn.PReLU()
                                    )

        #Block2: Residual_Blocks
        residual_blocks = []

        for _ in range(residual_count):
            residual_blocks.append(Generator_Residual_Block())

        self.block2 = nn.Sequential(*residual_blocks)

        
        #Block3
        self.block3 = nn.Sequential(
                                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64)
                                    )
        
        #Block4
        upsample_blocks = []
        for _ in range(self.upsample_counts):
            upsample_blocks.append(Generator_Upsample_Block())

        self.block4 = nn.Sequential(*upsample_blocks)

        #Block5
        self.block5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, x:torch.Tensor):

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out = torch.add(out1, out3)         #Residual Connection

        out = self.block4(out)
        out = self.block5(out)

        return (torch.tanh(out)+1)/2.       #Scaling values in range [0,1]





#2: Discriminator


class Discriminator(nn.Module):
    '''
    Input: Image of any size with n_channels = 3
    Output: Vector od size [Batch_size, 1]
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        #Block1
        self.block1 = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=1),
                                nn.LeakyReLU(negative_slope= 0.2)
                                    )

        #Block2
        self.block2 = Discriminator_Block(in_channels= 64, out_channels =64, stride=2)
        #Block3
        self.block3 = Discriminator_Block(in_channels= 64, out_channels =128, stride=1)
        #Block4
        self.block4 = Discriminator_Block(in_channels= 128, out_channels =128, stride=2)
        #Block5
        self.block5 = Discriminator_Block(in_channels= 128, out_channels =256, stride=1)
        #Block6
        self.block6 = Discriminator_Block(in_channels= 256, out_channels =256, stride=2)
        #Block7
        self.block7 = Discriminator_Block(in_channels= 256, out_channels =512, stride=1)
        #Block8
        self.block8 = Discriminator_Block(in_channels= 512, out_channels =512, stride=2)

        #Classification_Block
        self.classification_block = nn.Sequential(
                                            nn.AdaptiveAvgPool2d(1),
                                            nn.Conv2d(in_channels = 512, out_channels= 1024, kernel_size = 1),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(in_channels = 1024, out_channels= 1, kernel_size = 1),
                                                )


    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out) 
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.classification_block(out)

        out = (torch.tanh(out)+1)/2.

        return out.view(batch_size,1)


if __name__ == "__main__":

	netG = Generator()
	netD = Discriminator()
