"""
git clone https://github.com/yagays/kanjivg-radical
漢字データセット作成
部首情報とSVG画像から、32x32のグレースケール画像とマルチホットラベルを生成します。
"""

import json
import torch
import os
import re
import subprocess
import argparse
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

def load_element_data(element_file: str) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    """部首データを読み込み、IDマッピングとマルチホットベクトルを生成"""
    print(f" Loading element data ...")
    with open(element_file, "r", encoding="utf-8") as f:
        element2kanji = json.load(f)
    
    element_list = sorted(element2kanji.keys())
    element2id = {element: idx for idx, element in enumerate(element_list)}
    
    # 漢字→部首IDのマッピングを作成
    kanji2label: Dict[str, Set[int]] = {}
    for element, kanji_list in element2kanji.items():
        for kanji in kanji_list:
            if kanji not in kanji2label:
                kanji2label[kanji] = set()
            kanji2label[kanji].add(element2id[element])
    
    # マルチホットベクトルに変換
    num_elements = len(element2id)
    kanji2multihot = {}
    for kanji, element_id_set in kanji2label.items():
        vec = torch.zeros(num_elements)
        for eid in element_id_set:
            vec[eid] = 1.0
        kanji2multihot[kanji] = vec
    
    return element2id, kanji2multihot

def create_svg_code_mapping(svg_dir: str) -> Dict[str, str]:
    """SVGファイルから漢字とコードのマッピングを作成"""

    print(f" Making svg code mapping")  
    kanji2svgcode = {}
    for fname in os.listdir(svg_dir):
        if not fname.endswith(".svg"):
            continue
        
        code = os.path.splitext(fname)[0].lower()
        path = os.path.join(svg_dir, fname)
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        match = re.search(r'kvg:element="(.)"', content)
        if match:
            kanji = match.group(1)
            kanji2svgcode[kanji] = code
    
    return kanji2svgcode

def convert_svg_to_png(svg_path: str, png_path: str, size: Tuple[int, int] = (32, 32)):
    """SVGをPNGに変換し、リサイズ"""
    try:
        # 高解像度で一時的に変換
        temp_png = png_path.replace(".png", "_temp.png")
        subprocess.run(["magick", "-density", "1200", svg_path, temp_png], check=True)
        
        # リサイズとグレースケール変換
        img = Image.open(temp_png).convert("L")
        img = img.resize(size, Image.LANCZOS)
        img.save(png_path)
        
        os.remove(temp_png)
    except Exception as e:
        raise RuntimeError(f"Error converting {svg_path}: {e}")

def create_dataset(
    element_file: str,
    svg_dir: str,
    output_dir: str,
    output_json: str,
    skip_existing: bool = True
) -> None:
    """データセットの作成を実行
    
    Args:
        element_file: 部首データのJSONファイルパス
        svg_dir: SVGファイルのディレクトリパス
        output_dir: 出力PNG画像のディレクトリパス
        output_json: 出力JSONファイルのパス
        skip_existing: 既存のPNG画像がある場合にスキップするかどうか
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データの読み込み
    element2id, kanji2multihot = load_element_data(element_file)
    kanji2svgcode = create_svg_code_mapping(svg_dir)
    
    # 画像の変換とデータセットの作成
    final_dataset = {}
    if skip_existing:
        print(f" Warning, Skip Making png")
    elif skip_existing == False:
        print(f" Warning, Overwrite png")

    for kanji, multihot in tqdm(kanji2multihot.items(), desc="Processing images"):
        if kanji not in kanji2svgcode:
            print(f"⚠️ SVG code not found for {kanji}")
            continue

        code = kanji2svgcode[kanji]

        if skip_existing: # 既存のPNG画像がある場合はスキップ
            # print(f" Warning, Skip Making png")

            final_dataset[f"{code}.png"] = {
                    "label": multihot.tolist(),
                    "kanji": kanji
                }
            
        elif skip_existing == False: # 既存のPNG画像がない場合は作成
            # print(f" Warning, Overwrite png")
            svg_path = os.path.join(svg_dir, f"{code}.svg")
            png_path = os.path.join(output_dir, f"{code}.png")

            if not os.path.exists(svg_path):
                print(f"⚠️ SVG file not found: {svg_path}")
                continue
            try:
                convert_svg_to_png(svg_path, png_path)
                final_dataset[f"{code}.png"] = {
                    "label": multihot.tolist(),
                    "kanji": kanji
                }
            except Exception as e:
                print(f"❌ Error processing {kanji}: {e}")
    
    # データセットの保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print("✅ Dataset creation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="漢字データセット作成スクリプト")
    parser.add_argument("--element-file", type=str,default="I:/Kanji_dataset/kanjivg-radical/data/element2kanji.json",
                      help="部首データのJSONファイルパス")
    parser.add_argument("--svg-dir", type=str, default="I:/Kanji_dataset/kanjivg-radical/kanjivg-20160426-main/kanji",
                      help="SVGファイルのディレクトリパス")
    parser.add_argument("--output-dir", type=str, default="I:/Kanji_dataset/kanjivg-radical/kanji_image_data",
                      help="出力PNG画像のディレクトリパス")
    parser.add_argument("--output-json", type=str, default="I:/Kanji_dataset/kanjivg-radical/data/final_dataset_test.json",
                      help="出力JSONファイルのパス")
    parser.add_argument("--no_skip_existing",default=False, action="store_true",
                      help="既存のPNG画像がある場合でも上書きする")
    
    args = parser.parse_args()
    
    create_dataset(
        element_file=args.element_file,
        svg_dir=args.svg_dir,
        output_dir=args.output_dir,
        output_json=args.output_json,
        skip_existing=not args.no_skip_existing
    )


# # python make_kanji_dataset.py \
#     --element-file "I:/Kanji_dataset/kanjivg-radical/data/element2kanji.json" \
#     --svg-dir "I:/Kanji_dataset/kanjivg-radical/kanjivg-20160426-main/kanji" \
#     --output-dir "I:/Kanji_dataset/kanjivg-radical/kanji_image_data" \
#     --output-json "I:/Kanji_dataset/kanjivg-radical/data/final_dataset.json"