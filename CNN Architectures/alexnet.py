#importing the libraries
import torch
import torch.nn as nn



#Defining the model
class AlexNet(nn.Module):
    def __init__(self, n_outputs):
        super(AlexNet, self).__init__()

        #Number of layers on the output layer
        self.n_outputs = n_outputs

        #Block1
        self.block1 = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=4),
                            nn.ReLU(),
                            nn.BatchNorm2d(96),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                                    )
        
        #Block2
        self.block2 = nn.Sequential(
                            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.BatchNorm2d(256),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                                    )
        
        #Block3
        self.block3 = nn.Sequential(
                            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                            nn.ReLU()
                                    )
        
        #Block4
        self.block4 = nn.Sequential(
                            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                            nn.ReLU()
                                    )
        
        #Block5
        self.block5 = nn.Sequential(
                            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(256),
                            nn.MaxPool2d(kernel_size=3, stride=2)
                                    )
        
        
        #FCLayers: Classification
        self.classification_layers = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=1),
                            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
                            nn.Dropout2d(0.5),
                            nn.Conv2d(in_channels=4096, out_channels=self.n_outputs, kernel_size=1)
                                                    )
        
        

    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.classification_layers(out) #Output:[Batch_size, Channels, H, W]

        f_out = out.view(batch_size, -1)

        return f_out




if __name__ == '__main__':
	model = AlexNet(10)