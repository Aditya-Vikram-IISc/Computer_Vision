#importing the libraries
import torch
import torch.nn as nn


__all_models__ = ["VGG_Module", "VGG16", "VGG19"]



#Building Block of the VGG model
class VGG_Module(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, stride_kernel=1, n_modules=2):
        super(VGG_Module, self).__init__()

        #defining the input block
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size ,stride= stride_kernel, padding=1)

        #defining other layers in the module
        block_layers = []

        for _ in range(n_modules-1):
            block_layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1))
        
        #Adding the maxpool layers
        block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(*block_layers)

    def forward(self, x:torch.Tensor):
        out = self.block1(x)
        out = self.block2(out)

        return out



#Model1: VGG16

class VGG16(nn.Module):
    def __init__(self, input_channels=3, n_outputs=10):
        super(VGG16, self).__init__()


        #All blocks of the VGG16 model: Refer paper for more info
        self.block1 = VGG_Module(in_channels=3, out_channels=64,n_modules=2)
        self.block2 = VGG_Module(in_channels=64,out_channels=128,n_modules=2)
        self.block3 = VGG_Module(in_channels=128,out_channels=256,n_modules=3)
        self.block4 = VGG_Module(in_channels=256,out_channels=512,n_modules=3)
        self.block5 = VGG_Module(in_channels=512,out_channels=512,n_modules=3)

        #Defining the classification_head
        self.classification_head = nn.Sequential(
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(in_channels=512, out_channels=4096,kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4096, out_channels=4096,kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4096, out_channels=n_outputs, kernel_size=1)
                                                )
        
    def forward(self, x:torch.Tensor):

        batch_size = x.size(0)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.classification_head(out)
        out_f = out.view(batch_size, -1)
        return out_f


#Model2: VGG19

class VGG19(nn.Module):
    def __init__(self, input_channels=3, n_outputs=10):
        super(VGG19, self).__init__()


        #All blocks of the VGG16 model: Refer paper for more info
        self.block1 = VGG_Module(in_channels=3, out_channels=64,n_modules=2)
        self.block2 = VGG_Module(in_channels=64,out_channels=128,n_modules=2)
        self.block3 = VGG_Module(in_channels=128,out_channels=256,n_modules=4)
        self.block4 = VGG_Module(in_channels=256,out_channels=512,n_modules=4)
        self.block5 = VGG_Module(in_channels=512,out_channels=512,n_modules=4)

        #Defining the classification_head
        self.classification_head = nn.Sequential(
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(in_channels=512, out_channels=4096,kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4096, out_channels=4096,kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=4096, out_channels=n_outputs, kernel_size=1)
                                                )
        
    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.classification_head(out)
        f_out = out.view(batch_size, -1)

        return f_out


if __name__ = "__main__":
	model = VGG16()