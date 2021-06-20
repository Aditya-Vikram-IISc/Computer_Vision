import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


__all_losses__ = ["Content_Loss", "TV_Loss"]



class Content_loss(nn.Module):
    def __init__(self, layer_id:int=36):
        super(Content_loss, self).__init__()

        #Load pre-trained VGG19 model on ImageNet
        vgg19 = torchvision.models.vgg19(pretrained= True)

        #Extract the features from the pre-defined layer
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:layer_id]).eval()

        #Freeze the weighting updates
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        loss = F.mse_loss(self.feature_extractor(output), self.feature_extractor(target))

        return loss


class TV_Loss(nn.Module):
    def __init__(self, weight:float = 1.):
        super(TV_Loss, self).__init__()

        self.weight = weight

    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self.tensor_totalsize(x[:,:,1:,:])
        count_w = self.tensor_totalsize(x[:,:,:,1:])

        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x - 1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x - 1]),2).sum()

        loss = 2 * self.weight * (h_tv/count_h + w_tv/count_w) / batch_size
        return loss

    @staticmethod
    def tensor_totalsize(x:torch.Tensor):
        return x.size(1)*x.size(2)*x.size(3)
