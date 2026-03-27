#!/usr/bin/env python3
"""
第四篇论文完整预处理Pipeline
功能：从原始5波段影像 -> 拉伸 -> 分别生成真彩色和假彩色切片
参数：Gamma=1.68449, Percent Clip=0.25%-99.75%, 裁剪512x512，重叠20像素
"""

import os
import numpy as np
import rasterio
import cv2
from pathlib import Path
from tqdm import tqdm

# ====== 用户配置区域 ======
# 输入输出路径
INPUT_DIR = "/home/hxr/fourth_paper/data/original_data"  # 原始影像目录
OUTPUT_BASE = "/home/hxr/fourth_paper/data/processed_data"

# 真彩色输出目录
RGB_TILES_DIR = os.path.join(OUTPUT_BASE, "rgb_tiles")

# 假彩色输出目录
FC_TILES_DIR = os.path.join(OUTPUT_BASE, "fc_tiles")

# 预处理参数（与第三篇论文一致）
GAMMA = 1.68449
MIN_PERCENT = 0.25
MAX_PERCENT = 99.75

# 切片参数
TILE_SIZE = 512
OVERLAP = 20
STEP = TILE_SIZE - OVERLAP
MIN_TILE_SIZE = TILE_SIZE // 4  # 跳过小于1/4 tile_size的切片
# =========================


def percent_clip_stretch(arr, min_percent=MIN_PERCENT, max_percent=MAX_PERCENT, gamma=GAMMA):
    """
    Percent Clip + Gamma 拉伸（与第三篇论文一致）
    """
    # 移除 NaN 和无效值
    valid_mask = ~np.isnan(arr)
    valid_data = arr[valid_mask]

    if len(valid_data) == 0:
        return arr.astype(np.uint8)

    # 计算百分位数
    min_val = np.percentile(valid_data, min_percent)
    max_val = np.percentile(valid_data, max_percent)

    # 防止分母为0
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)

    # 将数据归一化到 [0, 1]
    normalized = (arr - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)

    # 应用伽马校正
    stretched = np.power(normalized, 1 / gamma)

    # 映射到 0~255
    result = (stretched * 255).astype(np.uint8)

    return result


def process_image(file_path):
    """
    处理单个图像文件：
    1. 读取5波段原始数据
    2. 应用 gamma 拉伸
    3. 分别生成真彩色(RGB)和假彩色(NIR-R-G)
    4. 裁剪成512x512切片
    """
    file_name = os.path.basename(file_path)

    try:
        with rasterio.open(file_path) as src:
            # 读取所有波段
            img_data = src.read()  # (C, H, W)

            # 假设波段顺序为：B(1), G(2), R(3), NIR(4), Cloud(5)
            # 转换为 (H, W, C) 格式
            img_hwc = np.transpose(img_data, (1, 2, 0))  # (H, W, C)

            # 应用 gamma 拉伸（只对前4个波段）
            img_stretched = np.zeros_like(img_hwc[:, :, :4], dtype=np.uint8)
            for i in range(4):
                img_stretched[:, :, i] = percent_clip_stretch(img_hwc[:, :, i])

            # 生成真彩色和假彩色
            # 真彩色：R(3), G(2), B(1) -> 索引2, 1, 0
            img_rgb = img_stretched[:, :, [2, 1, 0]]  # R, G, B

            # 假彩色：NIR(4), R(3), G(2) -> 索引3, 2, 1
            img_fc = img_stretched[:, :, [3, 2, 1]]  # NIR, R, G

            # 裁剪并保存切片
            rgb_count = slice_and_save(img_rgb, RGB_TILES_DIR, file_name.replace('.tif', '_rgb'))
            fc_count = slice_and_save(img_fc, FC_TILES_DIR, file_name.replace('.tif', '_fc'))

            return rgb_count, fc_count

    except Exception as e:
        print(f"处理失败 {file_name}: {e}")
        return 0, 0


def slice_and_save(img, output_dir, prefix):
    """
    将图像切成512×512的小块，重叠20像素
    使用坐标命名方式，方便后续转换回大图坐标系
    文件名格式: {prefix}_tile_{x}_{y}.png
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = img.shape[:2]
    count = 0

    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            # 提取切片
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # 跳过太小的边边角角（<1/4 tile_size）
            if tile.shape[0] < MIN_TILE_SIZE or tile.shape[1] < MIN_TILE_SIZE:
                continue

            # 如果切片小于目标尺寸，进行填充
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # 保存切片，使用坐标命名
            # 文件名格式: {prefix}_tile_{x}_{y}.png
            # x, y 是tile在大图中的起始坐标
            tile_name = f"{prefix}_tile_{x}_{y}.png"
            tile_path = os.path.join(output_dir, tile_name)

            # 转换RGB到BGR（OpenCV格式）
            cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

            count += 1

    return count


def main():
    print("===== 第四篇论文预处理Pipeline =====")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_BASE}")
    print(f"Gamma值: {GAMMA}")
    print(f"Percent Clip: {MIN_PERCENT}% - {MAX_PERCENT}%")
    print(f"切片参数: {TILE_SIZE}×{TILE_SIZE}, 重叠{OVERLAP}像素")
    print(f"步长: {STEP}像素")
    print(f"最小切片尺寸: {MIN_TILE_SIZE}像素")

    # 创建输出目录
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 获取所有tif文件
    tif_files = list(Path(INPUT_DIR).glob("*.tif"))
    print(f"\n找到 {len(tif_files)} 个tif文件")

    if len(tif_files) == 0:
        print("错误：没有找到tif文件！")
        return

    # 处理每个文件
    total_rgb_tiles = 0
    total_fc_tiles = 0
    success_count = 0

    for file_path in tqdm(tif_files, desc="处理文件"):
        rgb_count, fc_count = process_image(str(file_path))
        total_rgb_tiles += rgb_count
        total_fc_tiles += fc_count
        if rgb_count > 0 or fc_count > 0:
            success_count += 1

    print(f"\n===== 预处理完成 =====")
    print(f"成功处理: {success_count}/{len(tif_files)} 个文件")
    print(f"真彩色切片总数: {total_rgb_tiles}")
    print(f"假彩色切片总数: {total_fc_tiles}")
    print(f"\n输出目录:")
    print(f"  真彩色: {RGB_TILES_DIR}")
    print(f"  假彩色: {FC_TILES_DIR}")


if __name__ == "__main__":
    main()
