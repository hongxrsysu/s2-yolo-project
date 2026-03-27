# S2 YOLO Project

哨兵二号卫星图像船舶检测项目。使用YOLO模型对哨兵二号L1C数据进行预处理、推理和可视化。

## 项目结构

```
s2-yolo-project/
├── code/                    # 核心代码
│   ├── preprocess_final.py  # 哨兵二号原始数据预处理
│   ├── infer_final.py       # YOLO推理脚本
│   ├── vis_results.py       # 可视化结果脚本
│   ├── config.py            # GCP配置文件
│   └── requirements.txt     # Python依赖
├── weights/                 # YOLO权重文件
│   ├── rgb.pt              # 真彩色权重（~50MB）
│   └── fc.pt               # 假彩色权重（~150MB）
├── manifest/               # 索引文件（GCP上生成）
├── results/                # 处理结果（GCP上生成）
│   ├── rgb_vis/            # 真彩色可视化结果
│   ├── fc_vis/             # 假彩色可视化结果
│   ├── rgb_txt/            # 真彩色检测结果
│   └── fc_txt/             # 假彩色检测结果
└── logs/                   # 处理日志
```

## 快速开始

### 1. 环境安装

推荐使用 `uv` 快速安装环境：

```bash
# 安装uv（如果没有）
pip install uv

# 创建虚拟环境并安装依赖
cd s2-yolo-project/code
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 2. 配置

编辑 `code/config.py`，设置你的GCP配置：

- `USE_PUBLIC_DATA`: 是否使用公共哨兵二号数据桶
- `BUCKET`: 你的结果存储桶名称
- `REGION`: GCP区域
- `VM_WORK_DIR`: VM工作目录

### 3. 运行流程

#### 本地测试

```bash
# 预处理：生成切片
python code/preprocess_final.py

# 推理：运行YOLO检测
python code/infer_final.py

# 可视化：生成可视化结果（可选）
python code/vis_results.py
```

#### GCP上运行

1. 上传项目到GCP
2. 使用公共哨兵二号数据时，先在 `manifest/` 目录生成数据索引
3. 运行预处理和推理脚本

## 预处理参数

- **Gamma值**: 1.68449
- **Percent Clip**: 0.25% - 99.75%
- **切片大小**: 512×512 像素
- **重叠像素**: 20 像素

## 推理参数

- **图像尺寸**: 1280
- **真彩色置信度**: 0.1
- **假彩色置信度**: 0.1
- **IOU阈值**: 0.3

## 输出格式

检测结果txt文件格式：
```
class_id cx cy w h angle confidence
```

坐标已转换为大图坐标系，方便后续处理。

## 注意事项

- 权重文件较大，首次clone可能需要较长时间
- 在GCP上运行前，确保已配置好存储桶权限
- 建议使用GPU加速推理

## 许可证

本项目仅用于学术研究。
