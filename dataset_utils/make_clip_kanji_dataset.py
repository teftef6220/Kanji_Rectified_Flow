"""To Do バッチ処理にする"""


import json
import torch
import clip
from PIL import Image
from openai import OpenAI
import os
from tqdm import tqdm
from dotenv import load_dotenv

# OpenAI APIキー（必要なら環境変数や外部管理を推奨）
load_dotenv(verbose=True)
client = OpenAI(api_key = os.getenv("OPEN_AI_API_KEY"))

# CLIPモデルの準備
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# JSON読み込み
with open("I:/Kanji_dataset/kanjivg-radical/data/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_data = {}


# 各エントリに対して処理
for filename, info in tqdm(data.items(), desc="Processing"):
    kanji = info["kanji"]

    # GPTで意味を取得（例: GPT-3.5）
    response =  client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Explain the meaning of the Japanese kanji '{kanji}' in English in one sentence."}
        ]
    )
    explanation = response.choices[0].message.content

    print(explanation)

    # 説明をCLIPでベクトル化
    with torch.no_grad():
        text_tokens = clip.tokenize([explanation]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features[0].cpu().tolist()  # 1D list

    # 新しいデータとして保存
    output_data[filename] = {
        "kanji": kanji,
        "label": text_features
    }

# 保存
with open("I:/Kanji_dataset/kanjivg-radical/data/clip_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
