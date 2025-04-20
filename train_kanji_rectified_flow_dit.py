from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks.unet import UNet
from src.data_loader import KanjiDataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from torch.utils.data import Dataset
import json
import random
from networks.Dit import DiT, DiffuserCond



# from DiT_GumbelSoftmax.network.dit import DiffusionTransformer
# from DiT_GumbelSoftmax.config import Config

# parser
parser = ArgumentParser()
parser.add_argument("--optimizer", type=str, default="RAdamScheduleFree", help="optimizer")
parser.add_argument("--dataset", type=str, default="Kanji", help="dataset")
parser.add_argument("--train_use_cfg",type=bool,default=False,help="train with cfg")
parser.add_argument("--dropout_rate",type=float,default=0.1,help="dropout rate")
parser.add_argument("--cfg_scale",type=int,default=2.0,help="cfg scale")
parser.add_argument("--l1_norm",type=bool,default=False,help="l1 norm")
parser.add_argument("--use_dit",type=bool,default=False,help="use diffusion transformer")
parser.add_argument("--sampler",type=str,default="uniform",help="sampler. uniform or log sampler")
parser.add_argument("--seed",type=int,default=-1,help="seed")
parser.add_argument("--use_tqdm_discordbot",type=bool,default=False ,help="use tqdm discord bot")
parser.add_argument("--use_label_clip_embedding",type=bool,default=True,help="use label clip embedding")

args = parser.parse_args()

# Seed
if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
else:
    pass

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# model
if args.dataset == "Kanji":
    if args.use_label_clip_embedding:
        print("use label clip embedding")
        channel_num = 1
        data_label_num = 512

    else:
        channel_num = 1
        data_label_num = 1306


# dataloader 
if args.dataset == "Kanji":
    batch_size = 8
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 拡張して使うなら
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = KanjiDataLoader(
        data_dir="I:/Kanji_dataset/kanjivg-radical/kanji_image_data",
        json_path="I:/Kanji_dataset/kanjivg-radical/data/clip_embeddings.json",
        args=args,
        transform=transform,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs(f"result_{args.dataset}", exist_ok=True)

print("Training On ", device)
# train
epochs = 100

# use tqdm or not
if args.use_tqdm_discordbot:
    load_dotenv()
    from tqdm.contrib.discord import tqdm, trange
    token = os.environ["DISCORD_TOKEN"]
    channel_id = os.environ["CHANNEL_ID"]
    iterator = trange(epochs, token=token, channel_id=channel_id)
else:
    iterator = tqdm(range(epochs))
    # iterator = range(epochs)

criterion = torch.nn.MSELoss()

print("Use Optimizer: ", args.optimizer)

if args.optimizer == "adamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
elif args.optimizer == "RAdamScheduleFree":
    from schedulefree import RAdamScheduleFree
    optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
elif args.optimizer == "LBFGS":
    def closure(x,y):
        optimizer.zero_grad()               # init grad
        output = model(x)                   # Model
        loss = criterion(output, y)         # Loss
        loss.backward()                     # Grad
        return loss
    
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001)


total_loss_array,epoch_vec = [],[]

# for epoch in range(epochs):
for epoch in iterator:
    model.train()
    if args.optimizer == "RAdamScheduleFree":
        optimizer.train()
    total_loss = 0.0
    #tqdm
    # with tqdm(train_loader, unit="batch") as tepoch:
    # for i, (images, labels) in enumerate(train_loader):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        labels = labels.to(device)
        images = images.to(device) #(batch, 1, 28, 28)
        time = torch.rand(1).to(device)

        x_0 = torch.randn_like(images).to(device)
        x_t = time * images + (1 - time) * x_0


        if args.optimizer == "adamW" or args.optimizer == "RAdamScheduleFree":
            v_pred = model(x_t, time, labels)
            loss = criterion(images - x_0 , v_pred)

            if args.l1_norm:
                l1_lambda = 1e-5  # L1正則化の強度（適宜調整）
                l1_loss = 0.0
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                
                loss += l1_lambda * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        # tepoch.set_postfix(loss=loss.item())

    ave_loss = total_loss / len(train_loader) * batch_size
    print(f"Epoch: {epoch+1}, Loss: {ave_loss/(i+1)}")

    epoch_vec.append(epoch+1)
    total_loss_array.append(ave_loss/(i+1))

    plt.figure()
    plt.plot(epoch_vec, total_loss_array)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MSE Loss Plot")
    plt.savefig(f"result_{args.dataset}/recon_loss_plot.png")
    plt.close() 


    # check inference
    with torch.no_grad():
        model.eval()
        num_samples = 25  # 生成する画像の数

        if args.optimizer == "RAdamScheduleFree":
            optimizer.eval()
        if args.dataset == "Kanji":
            x_0 = torch.randn(num_samples, 1, 32, 32).to(device)

        # 時間ステップを適切に設定
        time = torch.linspace(0, 1, 10).to(device)  # 10個の時間ステップ
        time = time.unsqueeze(0).expand(num_samples, -1)  # (num_samples, 10)

        # テスト用のラベルを生成
        if args.use_label_clip_embedding:
            test_labels = torch.randn(num_samples, data_label_num).to(device)
        else:
            test_labels = torch.zeros((num_samples, data_label_num))
            for i in range(num_samples):
                num_active = random.randint(1, 5)
                indices = random.sample(range(data_label_num), num_active)
                test_labels[i, indices] = 1.0
            test_labels = test_labels.to(torch.float32).to(device)

        # 各時間ステップで推論
        for i in range(10):
            t = time[:, i]  # 現在の時間ステップ
            v_pred = model(x_0, t, test_labels)
            x_0 = x_0 + 0.1 * v_pred

        sample = (x_0 + 1) / 2
        sample.clamp_(0, 1)

        # 画像の保存処理
        pil_images = [transforms.functional.to_pil_image(x) for x in (sample * 255).to(torch.uint8)]
        cols, rows = 5, 5
        img_width, img_height = pil_images[0].size
        grid_width = img_width * cols
        grid_height = img_height * rows
        grid_image = Image.new("RGB", (grid_width, grid_height))

        for idx, img in enumerate(pil_images):
            x_offset = (idx % cols) * img_width
            y_offset = (idx // cols) * img_height
            grid_image.paste(img, (x_offset, y_offset))

        os.makedirs(f"result_{args.dataset}/images", exist_ok=True)
        grid_image.save(f"result_{args.dataset}/images/{epoch}_grid.png")

    if epoch % 20 == 0 and epoch != 0:
        os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
        torch.save(model.state_dict(), f"result_{args.dataset}/models/Dit_model_{epoch}_{args.dataset}_{args.sampler}_clip_{args.use_label_clip_embedding}.pth")


os.makedirs(f"result_{args.dataset}/models", exist_ok=True)
torch.save(model.state_dict(), f"result_{args.dataset}/models/Dit_model_last_{args.dataset}_{args.sampler}_clip_{args.use_label_clip_embedding}.pth")