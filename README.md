# SceneTalk-Splat

基于 3D 高斯泼溅（3D Gaussian Splatting）的场景理解系统，集成 YOLO 目标检测和 LLM 场景分析功能。

---

## 功能特性

- **YOLO 目标检测**：使用 YOLOv8 进行 2D 图像目标检测
- **3D 高斯渲染**：基于预训练的 3DGS 模型渲染场景
- **跨视角投影**：将 2D 检测框投影到 3D 空间
- **3D 物体重建**：从多视角检测结果重建 3D 物体
- **场景图谱构建**：分析物体间的空间关系
- **LLM 场景描述**：（可选）使用大语言模型生成场景描述
- **可视化报告**：生成 HTML 格式的分析报告

---

## 项目结构

```
SceneTalk-Splat/
├── main.py                 # 主程序入口
├── config/
│   └── config.yaml         # 配置文件
├── modules/                # 核心功能模块
│   ├── object_detection/   # YOLO 目标检测
│   ├── projection/         # 3D 投影与重建
│   ├── rendering/          # 高斯渲染
│   ├── scene_understanding/# 场景理解与 LLM
│   ├── visualization/      # 可视化
│   └── utils/              # 工具函数
├── gaussian_renderer/      # 高斯渲染器核心
├── scene/                  # 场景数据加载
├── arguments/              # 参数配置
├── utils/                  # 通用工具
├── submodules/             # 依赖子模块
│   ├── simple-knn/
│   └── diff-gaussian-rasterization/
└── requirements.txt        # Python 依赖
```

---

## 环境要求

- **操作系统**：Windows / Linux / macOS
- **Python**：3.8+
- **CUDA**：11.7+（推荐，用于 GPU 加速）
- **显存**：8GB+（推荐）

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone --recursive https://github.com/你的用户名/SceneTalk-Splat.git
cd SceneTalk-Splat
```

如果忘记 `--recursive`，需要手动初始化子模块：

```bash
git submodule init
git submodule update
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n scenetalk python=3.10
conda activate scenetalk

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. 安装 PyTorch

根据你的 CUDA 版本安装 PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch torchvision
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 编译子模块

```bash
# 安装 simple-knn
pip install submodules/simple-knn

# 安装 diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization
```

### 6. 下载 YOLO 模型

```bash
# 方法一：自动下载（首次运行时）
# 程序会自动从 Ultralytics 下载

# 方法二：手动下载
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt
```

---

## 数据准备

### 输入数据结构

```
data/your_scene/
├── images/           # 原始图像
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── sparse/           # COLMAP 稀疏重建结果
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── (可选) masks/     # 图像掩码
```

### 3DGS 模型

需要预先训练好的 3D Gaussian Splatting 模型：

```
output/your_scene/
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
└── cameras.json
```

> **提示**：可使用 [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 官方仓库训练模型。

---

## 配置文件

编辑 `config/config.yaml`：

```yaml
# 路径配置
paths:
  source_path: "data/your_scene"      # COLMAP 数据路径
  model_path: "output/your_scene"     # 3DGS 模型路径
  output_root: "output/analysis"      # 输出目录

# YOLO 配置
yolo:
  model_name: "yolov8x.pt"    # 模型: yolov8n/s/m/l/x
  conf_threshold: 0.25        # 置信度阈值
  device: "cuda"              # cuda 或 cpu

# LLM 配置（可选）
llm:
  enable_llm: false           # 是否启用 LLM
  provider: "openai"          # openai 或 anthropic
  api_key: "your-api-key"     # API 密钥
```

---

## 运行

### 基本用法

```bash
python main.py --config config/config.yaml
```

### 命令行参数

```bash
python main.py --config config/config.yaml \
               --source data/your_scene \
               --model output/your_scene \
               --output output/analysis
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config/config.yaml` |
| `--source` | 数据集路径（覆盖配置） | - |
| `--model` | 模型路径（覆盖配置） | - |
| `--output` | 输出目录（覆盖配置） | - |

---

## 输出结果

运行完成后，输出目录结构如下：

```
output/analysis/your_scene/
├── yolo/                    # YOLO 检测结果
│   ├── detections.json      # 检测数据
│   └── statistics.json      # 统计信息
├── yolo_vis/                # 检测可视化
│   └── 00000_detected.png
├── rendered_train/          # 渲染的训练视角
├── rendered_test/           # 渲染的测试视角
├── projected/               # 投影结果
├── projected_comparison/    # 对比图
├── scene_understanding/     # 场景理解
│   ├── object_database.json # 3D 物体数据库
│   └── scene_graph.json     # 场景图谱
├── report/                  # 分析报告
│   └── report.html          # HTML 报告
└── README.md                # 输出说明
```

---

## 处理流程

```
┌─────────────────┐
│   输入图像      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 1: YOLO    │ → 2D 目标检测
│ 目标检测        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 2: 高斯    │ → 渲染训练/测试视角
│ 场景渲染        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 3: 检测框  │ → 跨视角投影
│ 投影            │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 4: 3D      │ → 多视角聚合重建
│ 物体重建        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 5: 场景    │ → 空间关系分析
│ 理解            │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Step 6: 生成    │ → HTML 报告
│ 报告            │
└─────────────────┘
```

---

## API 使用

如果需要在代码中调用：

```python
from main import SceneUnderstandingPipeline

# 初始化
pipeline = SceneUnderstandingPipeline("config/config.yaml")

# 运行完整流程
pipeline.run()

# 或单独运行某个步骤
detection_results = pipeline.step1_yolo_detection()
pipeline.step2_gaussian_rendering()
```

---

## 常见问题

### 1. CUDA 内存不足

```bash
# 降低批处理大小或使用较小的 YOLO 模型
yolo:
  model_name: "yolov8s.pt"  # 使用小模型
```

### 2. 子模块编译失败

```bash
# 确保安装了正确版本的 CUDA 和编译工具
# Windows 需要 Visual Studio Build Tools
# Linux 需要 gcc/g++

# 重新编译
pip install submodules/simple-knn --force-reinstall
pip install submodules/diff-gaussian-rasterization --force-reinstall
```

### 3. 找不到图像

确保数据集结构正确，图像位于 `source_path/images/` 目录下。

### 4. LLM API 错误

```yaml
# 检查 API 密钥是否正确
llm:
  enable_llm: true
  api_key: "sk-your-actual-key"
```

---

## 依赖说明

| 依赖 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥2.0 | 深度学习框架 |
| ultralytics | ≥8.0 | YOLOv8 目标检测 |
| opencv-python | ≥4.8 | 图像处理 |
| open3d | ≥0.17 | 3D 数据处理 |
| openai | ≥1.3 | LLM API |
| loguru | ≥0.7 | 日志记录 |
| jinja2 | ≥3.1 | 报告生成 |

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{scenetalk-splat,
  title={SceneTalk-Splat: 3D Gaussian Scene Understanding System},
  author={Your Name},
  year={2024},
  url={https://github.com/你的用户名/SceneTalk-Splat}
}
```

本项目基于以下工作：

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 许可证

MIT License

---

## 联系方式

如有问题，请提交 [Issue](https://github.com/你的用户名/SceneTalk-Splat/issues)。
