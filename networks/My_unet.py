import torch as th
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import UNet

class My_Unet(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, time_embed_dim=256, cond_embed_dim=256, depth=4):
        super(My_Unet, self).__init__()

        self.unet = UNet(
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            cond_embed_dim=cond_embed_dim,
            time_embed_dim=time_embed_dim,
            depth=depth,
            kernel_size=[3,3,3,3,3,3,3],
            layers=[3,3,3,9,3,3,3],
            num_groups=[32] * (depth * 2 - 1) 
        )
    
    def forward(self, x, t, c):
        return self.unet(x, t, c)
    
class Time_Embed(nn.Module):
    def __init__(self, time_embed_dim=256):
        super(Time_Embed, self).__init__()
        self.time_embed = nn.Linear(1,time_embed_dim)
        self.reg = nn.ReLU(inplace=False)
    
    def forward(self, t):
        return self.reg(self.time_embed(t))

class Cond_Embed(nn.Module):
    def __init__(self,label_num=10, cond_embed_dim=256):
        super(Cond_Embed, self).__init__()
        self.hidden_layer = nn.Linear(label_num, 1024)
        self.cond_embed = nn.Linear(1024,cond_embed_dim)
        # self.normalize = nn.LayerNorm(cond_embed_dim)
        self.reg = nn.ReLU(inplace=False)
    
    def forward(self, c):
        c = self.hidden_layer(c)
        c = self.reg(c)
        out = self.cond_embed(c)
        # out = self.normalize(out)
        return out

class CombinedModel(nn.Module):
    def __init__(self, unet, time_embed, cond_embed):
        super(CombinedModel, self).__init__()
        self.unet = unet
        self.time_embed = time_embed
        self.cond_embed = cond_embed

    def forward(self, x, t, c):
        t = self.time_embed(t)
        c = self.cond_embed(c)
        return self.unet(x, t, c)