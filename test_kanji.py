from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks.unet import UNet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import random

# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="RAdamScheduleFree", help="optimizer")
parser.add_argument("--dataset", type=str, default="Kanji", help="dataset")
parser.add_argument("--gen_time_step", type=int, default=100, help="generate time step")
parser.add_argument("--gen_use_real_kanji_data",type=bool,default=True,help="generate use real kanji data")
parser.add_argument("--use_cfg",type=bool,default=False,help="train with cfg")
parser.add_argument("--cfg_scale",type=int,default=1.5,help="cfg scale")
parser.add_argument("--sampler",type=str,default="uniform",help="sampler . uniform or log_sampler. if you choose sampler, will load the model trained with the sampler in line 113")
parser.add_argument("--seed",type=int,default=-1,help="seed")

args = parser.parse_args()


class Unet(nn.Module):
    def __init__(self, in_channels, embedding_channels=64, time_embed_dim=256, cond_embed_dim=256, depth=4):
        super(Unet, self).__init__()

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
        self.cond_embed = nn.Linear(label_num,cond_embed_dim)
        # self.normalize = nn.LayerNorm(cond_embed_dim)
        self.reg = nn.ReLU(inplace=False)
    
    def forward(self, c):
        out = self.cond_embed(c)
        # out = self.normalize(out)
        return self.reg(out)


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


if __name__ == "__main__":

    if args.seed != -1:
        torch.manual_seed(args.seed)
    else:
        pass

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    unet = Unet(in_channels=1, embedding_channels=64).to(device)
    time_embed = Time_Embed().to(device)
    if args.dataset == "Kanji":
        label_num = 1306
        unet = Unet(in_channels=1, embedding_channels=64).to(device)
        time_embed = Time_Embed().to(device)
        cond_embed = Cond_Embed(label_num=label_num).to(device)

    model = CombinedModel(unet, time_embed, cond_embed).to(device)

    if args.optimizer == "RAdamScheduleFree":
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # load model
    pth_file_name = f"result_{args.dataset}/models/flow_model_40_{args.dataset}_{args.sampler}.pth"

    # load_model
    model.load_state_dict(torch.load(pth_file_name,map_location=device))
    with torch.no_grad():
        model.eval()
        if args.optimizer == "RAdamScheduleFree":
            optimizer.eval()
        if args.dataset == "Kanji":
            x_0 = torch.randn(25, 1, 32, 32).to(device)

        num_samples = 25 ## 適宜調整

        if args.gen_use_real_kanji_data: # 実際の漢字データを使用
            import json
            with open("I:/Kanji_dataset/kanjivg-radical/data/final_dataset.json", "r", encoding="utf-8") as f:
                kanji_data = json.load(f)
            # select random kanji from kanji_data .same label
            n = random.randint(0, len(kanji_data) - 1)
            keys = list(kanji_data.keys())
            nth_kanji_label = kanji_data[keys[n]]['label']
            kanji_name = kanji_data[keys[n]]["kanji"]
            
            # change nth_kanji_label to multi_hot_batch , list to tensor
            multi_hot_batch = torch.tensor(nth_kanji_label, dtype=torch.float32).to(device)

            print(f"generate {kanji_name}")

        else:
            kanji_label = random.randint(0, label_num - 1)
            multi_hot_batch = torch.zeros((num_samples, label_num))
            for i in range(num_samples):
                num_active = random.randint(10,12)  # 1〜n個の1を入れる
                indices = random.sample(range(label_num), num_active)
                multi_hot_batch[i, indices] = 1.0
            multi_hot_batch = multi_hot_batch.to(torch.float32).to(device)
            
        cond_embedded = cond_embed(multi_hot_batch).to(device) # torch.Size([256])
        
        # cond_embedded = cond_embed(nn.functional.one_hot(torch.arange(26), 26).float().cuda())
        
        if args.use_cfg:
            # uncond_labels = torch.zeros((cond_embedded.size(0), 26), device=device)
            uncond_labels = torch.zeros_like(multi_hot_batch, device=device)
            uncond_embedded = cond_embed(uncond_labels)
        
        if args.sampler == "uniform": ## 等間隔

            time_embedded = time_embed(torch.linspace(0, 1, args.gen_time_step).unsqueeze(1).to(device))
            for i in tqdm(range(args.gen_time_step)):
                v_cond = unet(x_0, time_embedded[i], cond_embedded)

                if args.use_cfg:
                    v_uncond = unet(x_0, time_embedded[i], uncond_embedded)
                    v = args.cfg_scale * v_cond - (args.cfg_scale - 1.) * v_uncond
                else:
                    v = v_cond

                x_0 = x_0 + (1/args.gen_time_step) * v
                    # x_0 = x_0.clamp(-1, 1)

        
        elif args.sampler == "log_sampler": 
                    
            log_steps = [args.gen_time_step]  # ここで args.gen_time_step = 1024 と仮定
            while log_steps[-1] > 1:
                next_step = max(1, log_steps[-1] // 2)
                log_steps.append(next_step)
            
            # log_steps は例: [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # 正規化して [1, 0.5, 0.25, 0.125, ... , 1/1024] のようにする
            time_steps = torch.tensor(log_steps, dtype=torch.float32, device=device) / log_steps[0]
            time_embedded = time_embed(time_steps.unsqueeze(1))

            # 以下、数値積分等の更新処理（注意：非等間隔更新の場合、単純な Euler 更新では誤差が大きくなる可能性がある）
            # dt の計算例（正規化した dt の合計が 1 になるように）
            raw_dts = torch.diff(time_steps)
            dts = raw_dts / raw_dts.sum()  # すべての dt の合計が 1 に

            prev_t = time_steps[0]
            dt_list = []
            for i in tqdm(range(len(time_steps) - 1)):
                v_cond = unet(x_0, time_embedded[i], cond_embedded)
                if args.use_cfg:
                    v_uncond = unet(x_0, time_embedded[i], uncond_embedded)
                    v = args.cfg_scale * v_cond - (args.cfg_scale - 1.) * v_uncond
                else:
                    v = v_cond
                dt = dts[i]
                dt_list.append(dt.item())
                x_0 = x_0 + dt * v
                prev_t = time_steps[i + 1]
            print(f"dt_list: {dt_list} , sum_dt: {sum(dt_list)}")



        sample = (x_0 + 1) / 2
        sample.clamp_(0, 1)

        pil_images = [transforms.functional.to_pil_image(x) for x in (sample * 255).to(torch.uint8)]
        #Save image in one image
        cols , rows = 5, 5
        img_width, img_height = pil_images[0].size
        grid_width = img_width * cols 
        grid_height = img_height * rows
        grid_image = Image.new("RGB", (grid_width, grid_height))

        for idx, img in enumerate(pil_images):
            x_offset = (idx % cols) * img_width  
            y_offset = (idx // cols) * img_height  
            grid_image.paste(img, (x_offset, y_offset))

        os.makedirs(f"result_{args.dataset}/test_images", exist_ok=True)
        grid_image.save(f"result_{args.dataset}/test_images/test_grid_{args.sampler}_ts_{args.gen_time_step}_seed_{args.seed}.png")

