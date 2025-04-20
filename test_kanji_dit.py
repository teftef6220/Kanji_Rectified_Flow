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
from networks.Dit import DiT, DiffuserCond

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
parser.add_argument("--use_label_clip_embedding",type=bool,default=True,help="use label clip embedding")

args = parser.parse_args()



if __name__ == "__main__":

    if args.seed != -1:
        torch.manual_seed(args.seed)
    else:
        pass

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.use_label_clip_embedding:
        label_num = 512
        # load clip
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        #  model path
        pth_file_name = f"result_{args.dataset}/models/Dit_model_60_{args.dataset}_{args.sampler}_clip_{args.use_label_clip_embedding}.pth"
    else:
        label_num = 1306
        # model path
        pth_file_name = f"result_{args.dataset}/models/Dit_model_last_{args.dataset}_{args.sampler}.pth"

    # モデルのハイパーパラメータ
    input_size = 32
    patch_size = 4
    in_channels = 1
    hidden_size = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    num_classes = 512 if args.use_label_clip_embedding else 1306  # Update num_classes based on label type
    learn_sigma = False

    model = DiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
    )
    model = model.to(device)  # Move model to device


    if args.optimizer == "RAdamScheduleFree":
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # load_model
    model.load_state_dict(torch.load(pth_file_name,map_location=device))
    with torch.no_grad():
        model.eval()
        num_samples = 25 ## 適宜調整

        if args.optimizer == "RAdamScheduleFree":
            optimizer.eval()

        #+---------------------
        # Make x_0 data (random noise : x_0 ~ N(0,1))
        #+---------------------
        x_0 = torch.randn(num_samples, 1, 32, 32).to(device)
        
        #+---------------------
        # Make cond_embed
        #+---------------------
        if args.use_label_clip_embedding: # clip label を使用する場合
            prompt = "Meaning of Instagram."
            with torch.no_grad():
                text_tokens = clip.tokenize([prompt]).to(device)
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features.float()  # float16 から float32 に変換
                text_features = text_features.expand(num_samples, -1)  # バッチサイズ分に拡張
        
        
        #+---------------------
        # Start generation
        #+---------------------
        if args.sampler == "uniform": # 等間隔
            time = torch.linspace(0, 1, args.gen_time_step).unsqueeze(1).to(device)
            # time_embedded = time_embed(torch.linspace(0, 1, args.gen_time_step).unsqueeze(1).to(device))
            for i in tqdm(range(args.gen_time_step)):
                v = model(x_0, time[i].expand(num_samples), text_features)  # timeの形状も修正

                x_0 = x_0 + (1/args.gen_time_step) * v
                    # x_0 = x_0.clamp(-1, 1)

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
        grid_image.save(f"result_{args.dataset}/test_images/Dit_test_grid_{args.sampler}_ts_{args.gen_time_step}_seed_{args.seed}.png")

