#!/usr/bin/env python3
"""
第四篇论文推理Pipeline（完整版）
使用不同的权重进行推理，并将坐标转换回大图坐标系
真彩色权重：/home/hxr/fourth_paper/weight/rgb.pt
假彩色权重：/home/hxr/fourth_paper/weight/fc.pt
输出：每个大图对应一个txt文件，包含所有检测结果
"""

import os
import cv2
import numpy as np
import re
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ====== 用户配置区域 ======
# 权重文件
RGB_WEIGHT_PATH = "/home/hxr/fourth_paper/weight/rgb.pt"
FC_WEIGHT_PATH = "/home/hxr/fourth_paper/weight/fc.pt"

# 输入数据目录
RGB_TILES_DIR = "/home/hxr/fourth_paper/data/processed_data/rgb_tiles"
FC_TILES_DIR = "/home/hxr/fourth_paper/data/processed_data/fc_tiles"

# 输出目录
OUTPUT_BASE = "/home/hxr/fourth_paper/inference_results"
RGB_VIS_DIR = os.path.join(OUTPUT_BASE, "rgb_vis")
FC_VIS_DIR = os.path.join(OUTPUT_BASE, "fc_vis")
RGB_TXT_DIR = os.path.join(OUTPUT_BASE, "rgb_txt")
FC_TXT_DIR = os.path.join(OUTPUT_BASE, "fc_txt")

# ====== 可视化开关 ======
# True: 生成可视化结果（较慢，但可以直接查看效果）
# False: 只生成txt检测结果（快速，适合大批量处理）
SAVE_VISUALIZATION = True
# =========================

# 推理参数
IMG_SIZE = 1280
CONF_TH_RGB = 0.1  # 真彩色置信度阈值
CONF_TH_FC = 0.1   # 假彩色置信度阈值
IOU_TH = 0.3
DEVICE = 0  # GPU

# 可视化参数
COLOR = (0, 255, 255)  # BGR 颜色
LINE_THICK = 1
FONT_SCALE = 0.45
FONT_THICK = 1
PAD_PX = 2
# =========================


def parse_tile_filename(tile_name):
    """
    解析切片文件名，提取大图名和坐标信息
    文件名格式: {prefix}_tile_{x}_{y}.png
    例如: S2_g279_2023-01-08_02-55-11_rgb_tile_0_0.png
    返回: (image_name, tile_x, tile_y)
    """
    # 匹配格式: xxx_tile_x_y.png
    pattern = r'^(.*)_tile_(\d+)_(\d+)\.png$'
    match = re.match(pattern, tile_name)

    if match:
        image_name = match.group(1)  # 大图名称
        tile_x = int(match.group(2))  # tile在大图中的x坐标
        tile_y = int(match.group(3))  # tile在大图中的y坐标
        return image_name, tile_x, tile_y
    else:
        return None, None, None


def draw_obb(im, xywhr_row, conf):
    """绘制旋转框"""
    xc, yc, w, h, ang = map(float, xywhr_row)
    w_pad = max(1.0, w + 2 * PAD_PX)
    h_pad = max(1.0, h + 2 * PAD_PX)
    ang_deg = ang * 180.0 / np.pi

    rect = ((xc, yc), (w_pad, h_pad), ang_deg)
    pts = cv2.boxPoints(rect).astype(int)
    cv2.polylines(im, [pts], True, COLOR, LINE_THICK)

    # 绘制置信度
    anchor = pts.min(axis=0)
    x, y = int(anchor[0]), int(anchor[1])
    cv2.putText(im, f"{conf:.2f}", (x, max(0, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, FONT_THICK, cv2.LINE_AA)


def convert_to_original_coords(xywhr, tile_x, tile_y):
    """
    将tile坐标系中的坐标转换为大图坐标系
    xywhr格式: [cx, cy, w, h, angle]
    返回: [cx_orig, cy_orig, w, h, angle]
    """
    cx, cy, w, h, ang = xywhr

    # 转换中心坐标到大图坐标系
    cx_orig = cx + tile_x
    cy_orig = cy + tile_y

    # 角度和尺寸保持不变
    return [cx_orig, cy_orig, w, h, ang]


def run_inference(model, tiles_dir, vis_output_dir, conf_th, data_type, save_vis=True):
    """
    对单个数据集进行推理
    返回: 按大图分组的检测结果 {image_name: [detections]}

    参数:
        save_vis: 是否保存可视化结果（默认True）
    """
    # 只在需要可视化时创建目录
    if save_vis:
        os.makedirs(vis_output_dir, exist_ok=True)

    # 获取所有切片文件
    tile_files = list(Path(tiles_dir).glob("*.png"))
    print(f"  找到 {len(tile_files)} 个{data_type}切片文件")

    # 按大图分组的检测结果
    # {image_name: [(cx, cy, w, h, angle, conf), ...]}
    detections_by_image = defaultdict(list)

    total_detections = 0
    total_confidence = 0.0
    detection_count = 0

    # 对每张切片进行推理
    for tile_file in tqdm(tile_files, desc=f"推理{data_type}", leave=False):
        img = cv2.imread(str(tile_file))
        if img is None:
            continue

        # 解析文件名，获取大图名和坐标
        tile_name = tile_file.name
        image_name, tile_x, tile_y = parse_tile_filename(tile_name)

        if image_name is None:
            print(f"  警告：无法解析文件名: {tile_name}")
            continue

        # 推理
        results = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=conf_th,
            iou=IOU_TH,
            device=DEVICE,
            save=False,
            show_labels=False,
            show_conf=False,
            verbose=False
        )

        # 处理结果
        # 只在需要可视化时复制图像
        im = img.copy() if save_vis else None
        if len(results) > 0:
            r = results[0]
            if getattr(r, "obb", None) is not None and r.obb is not None and len(r.obb) > 0:
                xywhr = r.obb.xywhr.cpu().numpy()
                confs = r.obb.conf.cpu().numpy().reshape(-1)

                for row, conf in zip(xywhr, confs):
                    # 只在需要可视化时绘制检测框
                    if save_vis and im is not None:
                        draw_obb(im, row, conf)

                    # 转换坐标到大图坐标系
                    orig_coords = convert_to_original_coords(row, tile_x, tile_y)

                    # 保存检测结果: (cx, cy, w, h, angle, conf)
                    detection = tuple(orig_coords + [conf])
                    detections_by_image[image_name].append(detection)

                    total_detections += 1
                    total_confidence += conf
                    detection_count += 1

        # 只在需要可视化时保存结果
        if save_vis and im is not None:
            vis_file = os.path.join(vis_output_dir, tile_name)
            cv2.imwrite(vis_file, im)

    # 计算统计信息
    avg_conf = total_confidence / detection_count if detection_count > 0 else 0.0

    stats = {
        "total_images": len(tile_files),
        "total_detections": total_detections,
        "avg_conf": avg_conf,
        "unique_large_images": len(detections_by_image)
    }

    print(f"  处理了 {stats['total_images']} 张{data_type}切片")
    print(f"  检测到 {stats['total_detections']} 个目标")
    print(f"  来自 {stats['unique_large_images']} 张大图")
    print(f"  平均置信度: {stats['avg_conf']:.3f}")

    return detections_by_image, stats


def save_detection_results(detections_by_image, output_txt_dir, data_type):
    """
    将检测结果保存为txt文件
    每张大图对应一个txt文件
    格式: class_id cx cy w h angle confidence
    """
    os.makedirs(output_txt_dir, exist_ok=True)

    # 为每张大图生成一个txt文件
    for image_name, detections in detections_by_image.items():
        txt_path = os.path.join(output_txt_dir, f"{image_name}.txt")

        with open(txt_path, 'w') as f:
            for detection in detections:
                # detection: (cx, cy, w, h, angle, conf)
                # class_id默认为0（船舶）
                cx, cy, w, h, angle, conf = detection
                f.write(f"0 {cx} {cy} {w} {h} {angle} {conf}\n")

    print(f"  已保存 {len(detections_by_image)} 个大图的检测结果到 {output_txt_dir}")


def main():
    print("===== 第四篇论文推理Pipeline（完整版）=====")
    print(f"真彩色权重: {RGB_WEIGHT_PATH}")
    print(f"假彩色权重: {FC_WEIGHT_PATH}")
    print(f"推理参数: imgsz={IMG_SIZE}, conf_rgb={CONF_TH_RGB}, conf_fc={CONF_TH_FC}, iou={IOU_TH}")

    # 检查权重文件
    if not os.path.exists(RGB_WEIGHT_PATH):
        print(f"错误：真彩色权重文件不存在！{RGB_WEIGHT_PATH}")
        return

    if not os.path.exists(FC_WEIGHT_PATH):
        print(f"错误：假彩色权重文件不存在！{FC_WEIGHT_PATH}")
        return

    # 检查输入数据
    if not os.path.exists(RGB_TILES_DIR):
        print(f"错误：真彩色数据不存在！{RGB_TILES_DIR}")
        print("请先运行预处理脚本: python preprocess_final.py")
        return

    if not os.path.exists(FC_TILES_DIR):
        print(f"错误：假彩色数据不存在！{FC_TILES_DIR}")
        print("请先运行预处理脚本: python preprocess_final.py")
        return

    # 加载模型
    print(f"\n加载模型...")
    rgb_model = YOLO(RGB_WEIGHT_PATH)
    fc_model = YOLO(FC_WEIGHT_PATH)

    # 创建输出目录
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"\n可视化开关: {SAVE_VISUALIZATION}")
    if SAVE_VISUALIZATION:
        print("  将生成可视化结果（较慢）")
    else:
        print("  只生成txt检测结果（快速）")

    # 推理真彩色数据
    print(f"\n--- 推理真彩色数据 ---")
    rgb_detections, rgb_stats = run_inference(
        rgb_model, RGB_TILES_DIR, RGB_VIS_DIR, CONF_TH_RGB, "真彩色", SAVE_VISUALIZATION
    )
    save_detection_results(rgb_detections, RGB_TXT_DIR, "真彩色")

    # 推理假彩色数据
    print(f"\n--- 推理假彩色数据 ---")
    fc_detections, fc_stats = run_inference(
        fc_model, FC_TILES_DIR, FC_VIS_DIR, CONF_TH_FC, "假彩色", SAVE_VISUALIZATION
    )
    save_detection_results(fc_detections, FC_TXT_DIR, "假彩色")

    # 打印总结
    print(f"\n===== 推理完成 =====")
    print(f"\n结果对比:")
    print(f"{'数据类型':<10} {'切片数量':<10} {'检测数量':<10} {'大图数量':<10} {'平均置信度':<15}")
    print("-" * 65)
    print(f"{'真彩色':<10} {rgb_stats['total_images']:<10} {rgb_stats['total_detections']:<10} {rgb_stats['unique_large_images']:<10} {rgb_stats['avg_conf']:<15.3f}")
    print(f"{'假彩色':<10} {fc_stats['total_images']:<10} {fc_stats['total_detections']:<10} {fc_stats['unique_large_images']:<10} {fc_stats['avg_conf']:<15.3f}")

    print(f"\n输出结果:")
    print(f"  真彩色检测结果(txt): {RGB_TXT_DIR}")
    print(f"  假彩色检测结果(txt): {FC_TXT_DIR}")
    if SAVE_VISUALIZATION:
        print(f"  真彩色可视化: {RGB_VIS_DIR}")
        print(f"  假彩色可视化: {FC_VIS_DIR}")
    else:
        print(f"  可视化结果未生成（如需可视化，请运行 vis_results.py 或设置 SAVE_VISUALIZATION=True）")

    print(f"\n💡 说明:")
    print(f"  - 每张大图对应一个txt文件，格式: image_name.txt")
    print(f"  - txt文件格式: class_id cx cy w h angle confidence")
    print(f"  - 坐标已转换为大图坐标系")
    print(f"  - class_id=0 表示船舶类")


if __name__ == "__main__":
    main()
