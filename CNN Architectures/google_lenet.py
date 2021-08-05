import torch
import torch.nn as nn


#No. of filters for each CNN layers in GoogleLeNet_Module. Pooling parameters defined separately
#Order: 1X1, (1X1, 3X3), (1X1,5X5), (MaxPool: not defined below, 1x1)

googlelenet_filters = {
                    "3a" : [64, 96, 128, 16, 32, 32],
                    "3b" : [128, 128, 192, 32, 96, 64],
                    "4a" : [192, 96, 208, 16, 48, 64],
                    "4b" : [160, 112, 224, 24, 64, 64],
                    "4c" : [128, 128, 256, 24, 64, 64],
                    "4d" : [112, 144, 288, 32, 64, 64],
                    "4e" : [256, 160, 320, 32, 128, 128],
                    "5a" : [256, 160, 320, 32, 128, 128],
                    "5b" : [384, 192, 384, 48, 128, 128],
                    "Maxpool3": [3,1,1],
                    "Maxpool4": [3,1,1],
                    "Maxpool5": [3,1,1],                    
                    }


#Building_Block
class GoogleLeNet_Block(nn.Module):
    def __init__(self, in_channels, filters_list, maxpool_list):
        super(GoogleLeNet_Block, self).__init__()

        #Defing the four branches of the inception block

        #Branch1: 1x1
        self.branch1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=filters_list[0], kernel_size=1, stride=1, padding=0),
                            nn.ReLU()
                                    )

        #Branch2: 1X1 ===> 3X3
        self.branch2 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=filters_list[1], kernel_size=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=filters_list[1], out_channels=filters_list[2], kernel_size=3, padding=1),
                            nn.ReLU()
                                    ) 
        
        #Branch3: 1x1 ===> 5x5
        self.branch3 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=filters_list[3], kernel_size=1, padding=0),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=filters_list[3], out_channels=filters_list[4], kernel_size=5, padding=2),
                            nn.ReLU()
                                    ) 
        
        #Branch4: MaxPool ===> 1X1
        self.branch4 = nn.Sequential(
                            nn.MaxPool2d(kernel_size= maxpool_list[0], stride= maxpool_list[1], padding=maxpool_list[2]),
                            nn.Conv2d(in_channels=in_channels, out_channels=filters_list[5], kernel_size=1, padding=0),
                            nn.ReLU()                           
                                    )
        
    def forward(self, x:torch.Tensor):

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out_concat = torch.cat([out1, out2, out3, out4], dim = 1)
        return out_concat


class GoogleLeNet(nn.Module):
    def __init__(self, in_channels:int=3, output_classes:int=1000, googlelenet_filters = googlelenet_filters):
        super(GoogleLeNet, self).__init__()


        #Block1:
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        
        #Block2:
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                    )
        
        #Block3: 2 inception block
        self.block3a = GoogleLeNet_Block(192, googlelenet_filters["3a"], googlelenet_filters["Maxpool3"])
        self.block3b = GoogleLeNet_Block(256, googlelenet_filters["3b"], googlelenet_filters["Maxpool3"])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        #Block4: 5 inception blocks
        self.block4a = GoogleLeNet_Block(480, googlelenet_filters["4a"], googlelenet_filters["Maxpool4"])
        self.block4b = GoogleLeNet_Block(512, googlelenet_filters["4b"], googlelenet_filters["Maxpool4"])
        self.block4c = GoogleLeNet_Block(512, googlelenet_filters["4c"], googlelenet_filters["Maxpool4"])
        self.block4d = GoogleLeNet_Block(512, googlelenet_filters["4d"], googlelenet_filters["Maxpool4"])
        self.block4e = GoogleLeNet_Block(528, googlelenet_filters["4e"], googlelenet_filters["Maxpool4"])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Block5: 2 inception block + 1 Avg pool
        self.block5a = GoogleLeNet_Block(832, googlelenet_filters["5a"], googlelenet_filters["Maxpool5"])
        self.block5b = GoogleLeNet_Block(832, googlelenet_filters["5b"], googlelenet_filters["Maxpool5"])
        self.avgpool5 = nn.AvgPool2d(kernel_size=7, stride=1)

        #Dropout
        self.dropout = nn.Dropout2d(0.5)

        #Classification_Head
        self.block6 = nn.Linear(in_features=1024, out_features= output_classes)


    def forward(self, x):
        batch_size = x.size(0)

        #input_size: (224,224,3)

        out = self.block1(x)
        out = self.block2(out)
        
        #3rd_Block
        out = self.block3a(out)
        out = self.block3b(out)
        out = self.maxpool3(out)

        #4th_Block
        out = self.block4a(out)
        out = self.block4b(out)
        out = self.block4c(out)
        out = self.block4d(out)
        out = self.block4e(out)
        out = self.maxpool4(out)

        #5th_Block
        out = self.block5a(out)
        out = self.block5b(out)
        out = self.avgpool5(out)
        out = self.dropout(out)

        #Reshaping for the classification_head
        out = out.view(batch_size, -1)

        #Passing it through the classification_head
        out = self.block6(out) 

        return out



if __name__ == "__main__":
	model = GoogleLeNet()

