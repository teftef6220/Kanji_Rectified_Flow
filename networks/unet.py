import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=256, cond_embed_dim=256, kernel_size=3, num_groups=1):
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.time_embed = nn.Linear(time_embed_dim, out_channels)
        self.cond_embed = nn.Linear(cond_embed_dim, out_channels)
        self.silu = nn.SiLU()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embed, cond_embed):
        r = self.gn1(x)
        r = self.silu(r)
        r = self.conv1(r)
        time_embed = self.silu(time_embed)
        time_embed = self.time_embed(time_embed)
        time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
        cond_embed = self.silu(cond_embed)
        cond_embed = self.cond_embed(cond_embed)
        cond_embed = cond_embed.unsqueeze(-1).unsqueeze(-1)
        r = r + time_embed + cond_embed
        r = self.gn2(r)
        r = self.silu(r)
        r = self.conv2(r)
        if hasattr(self, 'skip'):
            x = self.skip(x)
        x = x + r
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            embedding_channels = 64,
            cond_embed_dim=256,
            time_embed_dim=256,
            depth=3,
            kernel_size=[3,3,3,3,3],
            layers=[3,3,9,3,3],
            num_groups=[1,1,1,1,1]
        ):
        super().__init__()
        assert depth > 1
        assert len(kernel_size) == len(layers) == (depth * 2 - 1)

        self.in_conv = nn.Conv2d(in_channels, embedding_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        for i in range(depth-1):
            block = nn.ModuleList()
            if i != 0:
                block.append(Downsample(embedding_channels * 2**(i-1)))
                block.append(ResBlock(embedding_channels * 2**(i-1), embedding_channels * 2**i, time_embed_dim, cond_embed_dim, kernel_size[i], num_groups[i]))
            for _ in range(layers[i]):
                block.append(ResBlock(embedding_channels * 2**i, embedding_channels * 2**i, time_embed_dim, cond_embed_dim, kernel_size[i], num_groups[i]))
            self.downs.append(block)

        self.mid = nn.ModuleList()
        self.mid.append(Downsample(embedding_channels * 2**(depth-2)))
        self.mid.append(ResBlock(embedding_channels * 2**(depth-2), embedding_channels * 2**(depth-1), time_embed_dim, cond_embed_dim, kernel_size[depth-1], num_groups[depth-1]))
        for _ in range(layers[depth-1]):
            self.mid.append(ResBlock(embedding_channels * 2**(depth-1), embedding_channels * 2**(depth-1), time_embed_dim, cond_embed_dim, kernel_size[depth-1], num_groups[depth-1]))
        self.mid.append(Upsample(embedding_channels * 2**(depth-1)))
        self.mid.append(ResBlock(embedding_channels * 2**(depth-1), embedding_channels * 2**(depth-2), time_embed_dim, cond_embed_dim, kernel_size[depth-1], num_groups[depth-1]))

        self.ups = nn.ModuleList()
        for i in range(depth-1):
            block = nn.ModuleList()
            block.append(ResBlock(embedding_channels * 2**(depth-i-1), embedding_channels * 2**(depth-i-2), time_embed_dim, cond_embed_dim, kernel_size[depth+i], num_groups[depth+i]))
            for _ in range(layers[depth+i]):
                block.append(ResBlock(embedding_channels * 2**(depth-i-2), embedding_channels * 2**(depth-i-2), time_embed_dim, cond_embed_dim, kernel_size[depth+i], num_groups[depth+i]))
            if i is not depth-2:
                block.append(Upsample(embedding_channels * 2**(depth-i-2)))
                block.append(ResBlock(embedding_channels * 2**(depth-i-2), embedding_channels * 2**(depth-i-3), time_embed_dim, cond_embed_dim, kernel_size[depth+i], num_groups[depth+i]))
            self.ups.append(block)

        self.out = nn.Sequential(
            nn.GroupNorm(1, embedding_channels),
            nn.SiLU(),
            nn.Conv2d(embedding_channels, in_channels, kernel_size=3, padding=1),
        )
        
        self.out[2].weight.data.zero_()
        self.out[2].bias.data.zero_()

    def forward(self, x, time_embed, cond_embed):
        x = self.in_conv(x)
        skips = []
        for block in self.downs:
            for layer in block:
                if isinstance(layer, ResBlock):
                    x = layer(x, time_embed, cond_embed)
                elif isinstance(layer, Downsample):
                    x = layer(x)
            skips.append(x)
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                x = layer(x, time_embed, cond_embed)
            elif isinstance(layer, Upsample):
                x = layer(x)
            elif isinstance(layer, Downsample):
                x = layer(x)
        for block in self.ups:
            x = th.cat([x, skips.pop()], dim=1)
            for layer in block:
                if isinstance(layer, ResBlock):
                    x = layer(x, time_embed, cond_embed)
                elif isinstance(layer, Upsample):
                    x = layer(x)
        x = self.out(x)
        return x