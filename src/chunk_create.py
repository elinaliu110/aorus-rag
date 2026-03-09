#!/usr/bin/env python3
"""
chunk_create_zhen.py
將產品規格 CSV 轉換為結構化 chunks.json，包含：
  - common_spec    : 每個產品 × 每個規格欄位（排除 Video Graphics）
  - gpu_detail     : 每個產品的 GPU 詳細解析
  - product_summary: 每個產品的完整規格概覽 + 全系列比較摘要

Usage:
  python src/chunk_create_zhen.py \
      --input  data/specs.csv \
      --output data/chunks.json
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


SKIP_COMMON = {"Video Graphics"}


def extract_value(key: str, raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if key == "OS":
        return "、".join(l for l in lines if not l.startswith("*"))
    if key == "CPU":
        return lines[0] if lines else raw.strip()
    if key == "Display":
        return " ".join(l for l in lines if l)
    if key == "System Memory":
        return lines[0] if lines else raw.strip()
    if key == "Storage":
        return "，".join(l for l in lines if not l.startswith("*"))
    if key == "Keyboard Type":
        return raw.strip()
    if key == "Audio":
        return "，".join(l for l in lines if l)
    if key == "Communications":
        return "，".join(l for l in lines if l)
    if key == "Webcam":
        return "，".join(l for l in lines if l)
    if key == "Security":
        return raw.strip()
    if key in ("Battery", "Adapter", "Color"):
        return raw.strip()
    if key in ("Dimensions (W x D x H)", "Weight"):
        return lines[0] if lines else raw.strip()
    return raw.strip()


def parse_io_port(raw: str):
    left_items, right_items = [], []
    current_side = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("left side"):
            current_side = "left"
        elif line.lower().startswith("right side"):
            current_side = "right"
        elif current_side == "left":
            left_items.append(line)
        elif current_side == "right":
            right_items.append(line)

    extracted    = f"Left Side: {'；'.join(left_items)} / Right Side: {'；'.join(right_items)}"
    text_zh_body = f"左側包含 {'、'.join(left_items)}；右側包含 {'、'.join(right_items)}"
    text_en_body = f"Left Side: {', '.join(left_items)}; Right Side: {', '.join(right_items)}"
    return extracted, text_zh_body, text_en_body


def make_common_text(key: str, product: str, extracted: str, raw: str):
    if key == "OS":
        zh = f"{product} 支援的作業系統為 {extracted}。"
        en = f"{product} supports the following operating systems: {extracted}."
    elif key == "CPU":
        zh = f"{product} 搭載處理器 {extracted}。"
        en = f"{product} is equipped with the processor {extracted}."
    elif key == "Display":
        zh = f"{product} 的螢幕規格為 {extracted}。"
        en = f"{product} display specification: {extracted}."
    elif key == "System Memory":
        zh = f"{product} 的記憶體最大支援 {extracted}。"
        en = f"{product} supports up to 64GB DDR5 5600MHz system memory with 2x SO-DIMM slots."
    elif key == "Storage":
        zh = f"{product} 的儲存配置為 {extracted}。"
        en = f"{product} storage configuration: {extracted}."
    elif key == "Keyboard Type":
        zh = f"{product} 的鍵盤類型為 {extracted}。"
        en = f"{product} keyboard type: {extracted}."
    elif key == "I/O Port":
        _, text_zh_body, text_en_body = parse_io_port(raw)
        zh = f"{product} 的連接埠配置：{text_zh_body}。"
        en = f"{product} I/O port layout — {text_en_body}."
    elif key == "Audio":
        zh = f"{product} 的音訊規格為 {extracted}。"
        en = f"{product} audio features: {extracted}."
    elif key == "Communications":
        zh = f"{product} 的無線通訊支援 {extracted}。"
        en = f"{product} wireless connectivity supports WiFi 7 (802.11be 2x2), LAN 1G, Bluetooth 5.4."
    elif key == "Webcam":
        zh = f"{product} 的攝影機規格為 {extracted}。"
        en = f"{product} webcam specification: {extracted}."
    elif key == "Security":
        zh = f"{product} 的安全功能為 {extracted}。"
        en = f"{product} security feature: {extracted}."
    elif key == "Battery":
        zh = f"{product} 的電池容量為 {extracted}。"
        en = f"{product} battery capacity is {extracted}."
    elif key == "Adapter":
        zh = f"{product} 配備 {extracted} 變壓器。"
        en = f"{product} comes with a {extracted} power adapter."
    elif key == "Dimensions (W x D x H)":
        zh = f"{product} 的機身尺寸為 {extracted}。"
        en = f"{product} dimensions (W x D x H): {extracted}."
    elif key == "Weight":
        zh = f"{product} 的重量約為 {extracted}。"
        en = f"{product} weighs approximately {extracted}."
    elif key == "Color":
        zh = f"{product} 的顏色為 {extracted}。"
        en = f"{product} color: {extracted}."
    else:
        zh = f"{product} 的 {key} 為 {extracted}。"
        en = f"{product} {key}: {extracted}."
    return zh, en


def parse_gpu(product: str, short_id: str, raw: str) -> dict:
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and not l.strip().startswith("*")]

    model = re.sub(r"NVIDIA®\s*GeForce\s*", "", lines[0]).strip() if lines else ""

    vram_gb = vram_type = None
    for l in lines:
        m = re.match(r"(\d+)GB\s+(GDDR\d+)", l)
        if m:
            vram_gb, vram_type = int(m.group(1)), m.group(2)
            break

    tdp_w = None
    for l in lines:
        m = re.search(r"(\d+)W\s+Maximum", l)
        if m:
            tdp_w = int(m.group(1))
            break

    ai_boost_mhz = boost_clock_mhz = oc_mhz = None
    for l in lines:
        m = re.search(
            r"AI Boost\s*:\s*(\d+)\s*MHz\s*\((\d+)\s*MHz\s*Boost Clock\s*\+\s*(\d+)\s*MHz\s*OC\)", l
        )
        if m:
            ai_boost_mhz, boost_clock_mhz, oc_mhz = int(m.group(1)), int(m.group(2)), int(m.group(3))
            break

    text_zh = (
        f"{product} 的顯示卡為 {model}，"
        f"顯示記憶體 {vram_gb}GB {vram_type}，"
        f"最大圖形功耗 {tdp_w}W，"
        f"AI Boost 頻率 {ai_boost_mhz}MHz（Boost {boost_clock_mhz}MHz + OC {oc_mhz}MHz）。"
    )
    text_en = (
        f"{product} GPU: {model}, "
        f"{vram_gb}GB {vram_type}, "
        f"TDP {tdp_w}W, "
        f"AI Boost {ai_boost_mhz}MHz (Boost {boost_clock_mhz}MHz + OC {oc_mhz}MHz)."
    )
    return {
        "type": "gpu_detail",
        "product": product,
        "short_id": short_id,
        "key": "Video Graphics",
        "text": f"{text_zh} / {text_en}",
        "attributes": {
            "raw": raw.strip(),
            "model": model,
            "vram_gb": vram_gb,
            "vram_type": vram_type,
            "tdp_w": tdp_w,
            "ai_boost_mhz": ai_boost_mhz,
            "boost_clock_mhz": boost_clock_mhz,
            "oc_mhz": oc_mhz,
        },
        "text_zh": text_zh,
        "text_en": text_en,
    }


def make_product_summary(product: str, short_id: str, spec_map: dict, gpu_attr: dict) -> dict:
    gpu_model = gpu_attr.get("model", "")
    vram_gb   = gpu_attr.get("vram_gb", "")

    cpu_raw  = spec_map.get("CPU", "").strip()
    cpu_name = re.sub(r"\s*\(.*?\)", "", cpu_raw).strip()

    display_extracted = extract_value("Display", spec_map.get("Display", ""))
    memory_extracted  = extract_value("System Memory", spec_map.get("System Memory", ""))

    storage_lines = [l.strip() for l in spec_map.get("Storage", "").splitlines()
                     if l.strip() and not l.strip().startswith("*")]
    storage_max = next((l for l in storage_lines if "Up to" in l), "")

    battery_extracted = spec_map.get("Battery", "").strip()

    weight_lines = [l.strip() for l in spec_map.get("Weight", "").splitlines()
                    if l.strip() and not l.strip().startswith("*")]
    weight_extracted = weight_lines[0] if weight_lines else ""

    text_zh = (
        f"{product} 規格概覽："
        f"GPU {gpu_model} {vram_gb}GB VRAM，"
        f"CPU {cpu_name}，"
        f"螢幕 {display_extracted}，"
        f"記憶體 {memory_extracted}，"
        f"儲存最大 {storage_max}，"
        f"電池 {battery_extracted}，"
        f"重量 {weight_extracted}。"
    )
    text_en = (
        f"{product} overview: "
        f"GPU RTX {re.sub(r'RTX™ ', '', gpu_model)} {vram_gb}GB VRAM, "
        f"CPU Intel Core Ultra 9 275HX, "
        f"Display 16\" 16:10 OLED WQXGA 2560×1600 240Hz 1ms DCI-P3 100% 500nits NVIDIA G-SYNC Dolby Vision, "
        f"Memory up to 64GB DDR5 5600MHz, "
        f"Storage up to 4TB PCIe NVMe M.2 SSD, "
        f"Battery 99Wh, "
        f"Weight {weight_extracted}."
    )
    return {
        "type": "product_summary",
        "product": product,
        "short_id": short_id,
        "text": f"{text_zh} / {text_en}",
        "text_zh": text_zh,
        "text_en": text_en,
    }


def make_series_comparison(products: list, gpu_details: list) -> dict:
    series_name = "AORUS MASTER 16"
    model_names = [p.split()[-1] for p in products]
    gpu_parts_zh, gpu_parts_en = [], []
    for g in gpu_details:
        sid   = g["short_id"]
        model = g["attributes"]["model"]
        vram  = g["attributes"]["vram_gb"]
        tdp   = g["attributes"]["tdp_w"]
        gpu_parts_zh.append(f"{sid}（{model} {vram}GB VRAM，TDP {tdp}W）")
        gpu_parts_en.append(f"{sid} (RTX {re.sub(r'RTX™ ', '', model)}, {vram}GB VRAM, {tdp}W TDP)")

    top_model   = gpu_details[0]["short_id"]
    entry_model = gpu_details[-1]["short_id"]

    text_zh = (
        f"{series_name} 三款機型（{'、'.join(model_names)}）除 GPU 外規格完全相同。"
        f"GPU 差異：{'；'.join(gpu_parts_zh)}。"
        f"效能最強款為 {top_model}，搭載 VRAM 最大的 GPU。"
        f"入門款為 {entry_model}，VRAM 最小。"
        f"建議依 GPU 效能需求與預算選購。"
    )
    text_en = (
        f"{series_name} — Three models ({', '.join(model_names)}) share identical specs except GPU. "
        f"GPU differences: {'; '.join(gpu_parts_en)}. "
        f"Top performance: {top_model}. "
        f"Entry-level: {entry_model}. "
        f"Choose based on GPU needs and budget."
    )
    return {
        "type": "product_summary",
        "product": series_name,
        "short_id": "None",
        "text": f"{text_zh} / {text_en}",
        "text_zh": text_zh,
        "text_en": text_en,
    }


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
    products = [h.strip() for h in header[1:] if h.strip()]
    data = {p: {} for p in products}
    keys = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("Name", "").strip()
            if not key:
                continue
            keys.append(key)
            for p in products:
                data[p][key] = row.get(p, "").strip()
    return products, keys, data


def get_short_id(product: str) -> str:
    return product.split()[-1]


def build_chunks(products: list, keys: list, data: dict) -> list:
    chunks = []
    chunk_id = 0

    for product in products:
        short_id = get_short_id(product)
        for key in keys:
            if key in SKIP_COMMON:
                continue
            raw = data[product].get(key, "").strip()
            if not raw:
                continue
            extracted = extract_value(key, raw)
            if key == "I/O Port":
                extracted, _, _ = parse_io_port(raw)
            text_zh, text_en = make_common_text(key, product, extracted, raw)
            chunks.append({
                "id": chunk_id,
                "type": "common_spec",
                "short_id": short_id,
                "product": product,
                "key": key,
                "value": raw,
                "extracted": extracted,
                "text": f"{text_zh} / {text_en}",
                "text_zh": text_zh,
                "text_en": text_en,
            })
            chunk_id += 1

    gpu_details = []
    for product in products:
        short_id = get_short_id(product)
        raw_gpu = data[product].get("Video Graphics", "").strip()
        if not raw_gpu:
            continue
        chunk = parse_gpu(product, short_id, raw_gpu)
        chunk["id"] = chunk_id
        chunks.append(chunk)
        gpu_details.append(chunk)
        chunk_id += 1

    for product in products:
        short_id = get_short_id(product)
        gpu_attr = next(
            (g["attributes"] for g in gpu_details if g["product"] == product), {}
        )
        chunk = make_product_summary(product, short_id, data[product], gpu_attr)
        chunk["id"] = chunk_id
        chunks.append(chunk)
        chunk_id += 1

    series_chunk = make_series_comparison(products, gpu_details)
    series_chunk["id"] = chunk_id
    chunks.append(series_chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Convert product spec CSV to chunks.json"
    )
    parser.add_argument("--input",  "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading: {input_path}")
    products, keys, data = load_csv(str(input_path))
    print(f"[INFO] Found {len(products)} products × {len(keys)} spec keys")

    chunks = build_chunks(products, keys, data)
    print(f"[INFO] Generated {len(chunks)} chunks")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()
