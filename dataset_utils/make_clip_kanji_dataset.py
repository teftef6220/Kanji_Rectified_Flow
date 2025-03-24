import json
import torch
import clip
from PIL import Image
from openai import OpenAI
import os
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
import google.generativeai as genai

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Kanji", help="dataset")
parser.add_argument("--AI_model", type=str, default="gemini-1.5-flash", help="AI model")
args = parser.parse_args()

if args.AI_model == "gpt-4o-mini":
    load_dotenv(verbose=True)
    client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

elif args.AI_model == "gemini-1.5-flash":
    load_dotenv(verbose=True)
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 入力と保存ファイル
input_json_path = "I:/Kanji_dataset/kanjivg-radical/data/final_dataset.json"
output_json_path = "I:/Kanji_dataset/kanjivg-radical/data/clip_embeddings.json"

# データ読み込み
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 既存出力があれば読み込む（再開用）
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)
else:
    output_data = {}

# 全データループ
for filename, info in tqdm(data.items(), desc="Processing"):
    if filename in output_data:
        continue  # 既に処理済みならスキップ

    kanji = info["kanji"]
    multi_hot_label = info["label"]

    try:
        # 意味生成
        if args.AI_model == "gpt-4o-mini":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Explain the meaning of the Japanese kanji '{kanji}' in English in one sentence."}]
            )
            explanation = response.choices[0].message.content

        elif args.AI_model == "gemini-1.5-flash":
            question = f"Explain the meaning of the Japanese kanji '{kanji}' in English in one sentence."
            response = gemini_model.generate_content(question)
            explanation = response.text

        # CLIP埋め込み
        with torch.no_grad():
            text_tokens = clip.tokenize([explanation]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features[0].cpu().tolist()

        # 保存
        output_data[filename] = {
            "kanji": kanji,
            "label": multi_hot_label,
            "clip_embedding": text_features
        }

        # 一件ごとに保存（安心）
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue
