import torch
import torch.nn as nn



class MobileNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(MobileNet_block, self).__init__()

        #Depthwise separable CNN has 2 parts: 
        #### depthwise cnn for capturing spatial interaction
        #### pointwise cnn for capturing channel interaction

        self.depthwise_cnn = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU()
                                        )
        
        self.pointwise_cnn = nn.Sequential(
                                        nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU()
                                        )

    def forward(self, x:torch.Tensor):
        out = self.depthwise_cnn(x)
        out = self.pointwise_cnn(out)

        return out



class MobileNet(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=1000):
        super(MobileNet, self).__init__()

        #Block1
        self.block1 = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU()
                                    )
        
        #Block2
        self.block2 = MobileNet_block(in_channels=32, out_channels=64, stride=1, padding=1)

        #Block3
        self.block3a = MobileNet_block(in_channels=64, out_channels=128, stride=2, padding=1)
        self.block3b = MobileNet_block(in_channels=128, out_channels=128, stride=1, padding=1)

        #Block4
        self.block4a = MobileNet_block(in_channels=128, out_channels=256, stride=2, padding=1)
        self.block4b = MobileNet_block(in_channels=256, out_channels=256, stride=1, padding=1)

        #Block5
        self.block5a = MobileNet_block(in_channels=256, out_channels=512, stride=2, padding=1)

        block5b = []
        for _ in range(5):
            block5b.append(MobileNet_block(in_channels=512, out_channels=512, stride=1, padding=1))
        
        self.block5b = nn.Sequential(*block5b)

        
        #Block6
        self.block6a = MobileNet_block(in_channels=512, out_channels=1024, stride=2, padding=1)
        self.block6b = MobileNet_block(in_channels=1024, out_channels=1024, stride=1, padding=1)


        #Block7: Classification_Head
        self.block7 = nn.Sequential(
                            nn.AvgPool2d(kernel_size=7, stride=1),
                            nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1, stride=1)
                                    )


    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3a(out)
        out = self.block3b(out)
        out = self.block4a(out)
        out = self.block4b(out)
        out = self.block5a(out)
        out = self.block5b(out)
        out = self.block6a(out)
        out = self.block6b(out)
        out = self.block7(out) #Output: [Batch, Channels (1000), 1, 1 ]

        #Reshaping it to [Batch, Num_Classes]
        out_f = out.view(batch_size, -1)
        return out_f



if __name__ == "__main__":
	model = MobileNet()