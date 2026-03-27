#!/usr/bin/env python3
"""
第四篇论文可视化脚本
功能：读取txt检测结果，生成可视化图像
独立于推理脚本，可以单独运行
"""

import os
import cv2
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# ====== 用户配置区域 ======
# 检测结果目录
RGB_TXT_DIR = "/home/hxr/fourth_paper/inference_results/rgb_txt"
FC_TXT_DIR = "/home/hxr/fourth_paper/inference_results/fc_txt"

# 切片数据目录（用于读取原图）
RGB_TILES_DIR = "/home/hxr/fourth_paper/data/processed_data/rgb_tiles"
FC_TILES_DIR = "/home/hxr/fourth_paper/data/processed_data/fc_tiles"

# 可视化输出目录
RGB_VIS_OUTPUT = "/home/hxr/fourth_paper/inference_results/rgb_vis"
FC_VIS_OUTPUT = "/home/hxr/fourth_paper/inference_results/fc_vis"

# 可视化参数
COLOR = (0, 255, 255)  # BGR 颜色（青色）
LINE_THICK = 2
FONT_SCALE = 0.6
FONT_THICK = 2
PAD_PX = 3
SHOW_CONF = True  # 是否显示置信度
# =========================


def parse_tile_filename(tile_name):
    """
    解析切片文件名，提取大图名和坐标信息
    文件名格式: {prefix}_tile_{x}_{y}.png
    返回: (image_name, tile_x, tile_y)
    """
    pattern = r'^(.*)_tile_(\d+)_(\d+)\.png$'
    match = re.match(pattern, tile_name)
    if match:
        image_name = match.group(1)
        tile_x = int(match.group(2))
        tile_y = int(match.group(3))
        return image_name, tile_x, tile_y
    else:
        return None, None, None


def draw_obb(im, cx, cy, w, h, angle, conf):
    """
    在图像上绘制旋转框
    坐标格式: cx, cy (中心点), w, h (宽高), angle (角度)
    """
    # 膨胀框的大小
    w_pad = max(1.0, w + 2 * PAD_PX)
    h_pad = max(1.0, h + 2 * PAD_PX)
    ang_deg = angle * 180.0 / np.pi

    # 创建旋转矩形
    rect = ((cx, cy), (w_pad, h_pad), ang_deg)
    pts = cv2.boxPoints(rect).astype(int)
    cv2.polylines(im, [pts], True, COLOR, LINE_THICK)

    # 绘制置信度
    if SHOW_CONF:
        anchor = pts.min(axis=0)
        x, y = int(anchor[0]), int(anchor[1])
        cv2.putText(im, f"{conf:.2f}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, FONT_THICK, cv2.LINE_AA)


def read_detections_from_txt(txt_path):
    """
    从txt文件读取检测结果
    格式: class_id cx cy w h angle confidence
    返回: [(cx, cy, w, h, angle, conf), ...]
    """
    detections = []
    if not os.path.exists(txt_path):
        return detections

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                # class_id = int(parts[0])  # 不需要class_id，都是船舶
                cx, cy, w, h, angle, conf = map(float, parts[1:])
                detections.append((cx, cy, w, h, angle, conf))

    return detections


def visualize_data_type(txt_dir, tiles_dir, vis_output_dir, data_type):
    """
    可视化单个数据类型（真彩色或假彩色）
    """
    os.makedirs(vis_output_dir, exist_ok=True)

    # 获取所有txt文件
    txt_files = list(Path(txt_dir).glob("*.txt"))
    print(f"找到 {len(txt_files)} 个{data_type}检测结果文件")

    if len(txt_files) == 0:
        print(f"警告：{txt_dir} 目录中没有txt文件！")
        return

    total_visualized = 0
    total_detections = 0

    # 处理每个txt文件
    for txt_file in tqdm(txt_files, desc=f"可视化{data_type}"):
        # 读取检测结果
        detections = read_detections_from_txt(str(txt_file))

        if len(detections) == 0:
            continue

        # 获取对应的大图名称
        image_name = txt_file.stem  # 去掉.txt后缀

        # 查找所有相关的tile文件
        # 例如: image_name可能是 S2_g279_2023-01-08_02-55-11_rgb
        # 需要匹配 S2_g279_2023-01-08_02-55-11_rgb_tile_*.png
        tile_pattern = f"{image_name}_tile_*.png"
        tile_files = list(Path(tiles_dir).glob(tile_pattern))

        if len(tile_files) == 0:
            print(f"  警告：找不到 {image_name} 的切片文件")
            continue

        # 处理每个tile
        for tile_file in tile_files:
            # 读取tile图像
            img = cv2.imread(str(tile_file))
            if img is None:
                continue

            # 解析tile文件名，获取坐标
            tile_name = tile_file.name
            _, tile_x, tile_y = parse_tile_filename(tile_name)

            if tile_x is None:
                continue

            # 在该tile上绘制检测结果
            tile_detections = 0
            for detection in detections:
                cx, cy, w, h, angle, conf = detection

                # 检查检测框是否在这个tile内
                # 将大图坐标转换为tile坐标
                cx_tile = cx - tile_x
                cy_tile = cy - tile_y

                # 检查中心点是否在tile范围内（考虑一些边界情况）
                if -w/2 <= cx_tile <= 512 + w/2 and -h/2 <= cy_tile <= 512 + h/2:
                    draw_obb(img, cx_tile, cy_tile, w, h, angle, conf)
                    tile_detections += 1

            # 保存可视化结果
            vis_path = os.path.join(vis_output_dir, tile_name)
            cv2.imwrite(vis_path, img)
            total_visualized += 1
            total_detections += tile_detections

    print(f"  可视化了 {total_visualized} 个切片")
    print(f"  绘制了 {total_detections} 个检测框")


def main():
    print("===== 第四篇论文可视化Pipeline =====")
    print(f"真彩色检测结果: {RGB_TXT_DIR}")
    print(f"假彩色检测结果: {FC_TXT_DIR}")
    print(f"可视化参数: color={COLOR}, line_thickness={LINE_THICK}")
    print(f"显示置信度: {SHOW_CONF}")

    # 检查输入数据
    if not os.path.exists(RGB_TXT_DIR):
        print(f"错误：真彩色检测结果不存在！{RGB_TXT_DIR}")
        print("请先运行推理脚本: python infer_final.py")
        return

    if not os.path.exists(FC_TXT_DIR):
        print(f"错误：假彩色检测结果不存在！{FC_TXT_DIR}")
        print("请先运行推理脚本: python infer_final.py")
        return

    # 可视化真彩色结果
    print(f"\n--- 可视化真彩色结果 ---")
    visualize_data_type(RGB_TXT_DIR, RGB_TILES_DIR, RGB_VIS_OUTPUT, "真彩色")

    # 可视化假彩色结果
    print(f"\n--- 可视化假彩色结果 ---")
    visualize_data_type(FC_TXT_DIR, FC_TILES_DIR, FC_VIS_OUTPUT, "假彩色")

    print(f"\n===== 可视化完成 =====")
    print(f"真彩色可视化结果: {RGB_VIS_OUTPUT}")
    print(f"假彩色可视化结果: {FC_VIS_OUTPUT}")


if __name__ == "__main__":
    main()
