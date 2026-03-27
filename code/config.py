# GCP配置
# 是否使用公共哨兵二号数据
USE_PUBLIC_DATA = True  # True: 使用公共数据桶, False: 使用私有桶

BUCKET = "gs://s2-yolo-results-001"  # 自己的结果桶
REGION = "us-central1"  # 存储桶区域
VM_WORK_DIR = "/mnt/work"  # VM上的临时工作目录（Persistent Disk挂载点）
PUBLIC_L1C_BUCKET = "gs://gcp-public-data-sentinel-2/"  # 公共哨兵二号数据桶

# 路径映射
INPUT_DIR = f"{PUBLIC_L1C_BUCKET}tiles/" if USE_PUBLIC_DATA else f"{BUCKET}/original_data/"
OUTPUT_BASE = f"{VM_WORK_DIR}/processed_data"
RGB_TILES_DIR = f"{OUTPUT_BASE}/rgb_tiles"
FC_TILES_DIR = f"{OUTPUT_BASE}/fc_tiles"

# 权重文件路径（相对于项目根目录）
WEIGHTS_DIR = "../weights"
RGB_WEIGHT_PATH = f"{WEIGHTS_DIR}/rgb.pt"
FC_WEIGHT_PATH = f"{WEIGHTS_DIR}/fc.pt"
