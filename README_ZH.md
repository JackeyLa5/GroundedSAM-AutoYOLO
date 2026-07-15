# AutoYOLO

[English](README.md) | 简体中文

<p align="center">
  <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python">
  <img src="https://img.shields.io/badge/Node.js-22+-green" alt="Node.js">
  <img src="https://img.shields.io/badge/Platform-Ubuntu%20x86-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-orange" alt="GPU">
  <img src="https://img.shields.io/github/stars/JackeyLa5/GroundedSAM-AutoYOLO?style=social" alt="Stars">
</p>

```
🖼️ 图片/视频 → 🔍 Grounded-SAM 检测 → 🎯 实例分割 → ✏️ 修正 → 📦 导出 → 🚀 YOLO → ✅ 模型
```

**图片/视频扔进去 → YOLO 模型训出来**，基于 GroundingDINO + SAM2 的 Grounded-SAM 自动标注与人工闭环修正。多格式数据集导入导出、训练任务队列、一键 YOLO 训练（检测 & 分割）、视频关键帧提取、模型验证均面向 Ubuntu x86 + NVIDIA CUDA 的 Docker 部署。

本项目参考了原始项目 [VLM-AutoYOLO](https://github.com/Somnusochi/VLM-AutoYOLO) 的整体流程。当前版本的主要变化是将原来的 VLM 后端替换为 Grounded-SAM / GroundingDINO + SAM2，以降低显存需求，同时保留自动生成 YOLO 数据集与训练的工作流。

![业务流程](docs/architecture_zh.webp)

> 详细 Mermaid 图表请查看 [架构与流程文档](docs/architecture_diagram.md)

## 核心功能
- 🤖 **Grounded-SAM 自动标注**：GroundingDINO 开放词汇文本检测与 SAM2 实例分割
- 🎯 **实例分割**：在同一 Grounded-SAM 流程中输出 BBox 与像素级 mask，画布可独立开关
- 🎥 **视频标注**：智能关键帧提取（场景/运动/间隔），SSIM 去重
- ✏️ **人工修正**：Canvas 画框模式，NMS 过滤，单框隐藏
- 📦 **多格式导入/导出**：YOLO、YOLO-Seg、COCO JSON、Pascal VOC XML、CreateML JSON — 分片上传支持最大 10GB，断点续传
- 🚀 **训练队列**：任务排队串行执行，支持取消，一键 YOLO 训练（v8 / v11 / v26）SSE 实时进度
- ✅ **模型验证**：批量图片/视频测试，MJPEG 实时流，SSE 视频推理
- 💾 **智能模型管理**：Grounded-SAM 惰性加载、闲置自动卸载和 SSE 状态推送，CUDA 内存回收
- 🌐 **国际化**：中文 / English / 日本語 · 🎨 **主题**：亮色/暗色模式

## 面向视觉数据管线

AutoYOLO 专为闭环视觉数据流程设计：从摄像头、视频或已有图片数据采集图像，用 Grounded-SAM 自动标注，在浏览器中审核/修正标注结果，再训练出 YOLO 检测或分割模型——全流程无需离开应用。

## 文档

📚 **[用户指南 (中文)](docs/guide/README.md)** | 📚 **[User Guide (English)](docs/guide/en/README.md)**

完整指南：快速开始、标注最佳实践、训练参数调优、模型部署。

## 截图

| Grounded-SAM 预标注与人工修正 | YOLO 训练 |
|------------------|-----------|
| ![Grounded-SAM 预标注与人工修正](docs/1.webp) | ![YOLO 训练](docs/2.webp) |

| 视频关键帧入口 | 模型验证 |
|--------------|---------|
| ![视频关键帧入口](docs/4.webp) | ![模型验证](docs/3.webp) |

## 技术栈

| 层 | 技术 |
|---|------|
| 检测与分割 | GroundingDINO + SAM2（Grounded-SAM） |
| 目标检测 | YOLOv8 / v11 / v26 — 检测 & 分割（Ultralytics） |
| 后端 | Python FastAPI + PostgreSQL + SSE |
| 前端 | React + TypeScript + Vite + Tailwind CSS + antd |
| GPU 内存 | CUDA expandable segments / empty_cache |
| 状态管理 | Zustand + TanStack Query + ahooks |
| 国际化 | i18next（中文 / 英文 / 日本語） |
| 视频处理 | ffmpeg（场景检测 / 运动检测 / 间隔提取） |
| 工程化 | pnpm、ESLint、Prettier、Husky、commitlint、Playwright |

## 快速开始

### Docker 部署

> **环境要求：** Ubuntu x86_64、NVIDIA GPU、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、Docker、Node.js 22+ / pnpm。

直接执行这一条：

```bash
git clone https://github.com/JackeyLa5/GroundedSAM-AutoYOLO.git && cd GroundedSAM-AutoYOLO && cd frontend && pnpm install && pnpm build && cd .. && docker compose -f docker/docker-compose.yml up -d --build db backend frontend
```

打开：

```bash
http://localhost
```

**服务：**

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 | 80 | React 界面（Nginx） |
| 后端 | 8000 | FastAPI 服务器 |
| Grounded-SAM | 8002 | Grounded-SAM 独立推理服务 |
| 数据库 | 5432 | PostgreSQL |

**GPU 支持** — `docker/docker-compose.yml` 已内置 NVIDIA GPU 透传配置，后端镜像默认安装 Grounded-SAM2 依赖。

**持久化存储（Docker 卷）：**
- `pgdata` — 数据库 · `hf-cache` — Grounded-SAM 使用的 Hugging Face 缓存 · `uploads` — 用户上传 · `training-data` — 训练输出

**备份/恢复：**

```bash
docker compose -f docker/docker-compose.yml exec db pg_dump -U postgres autolabeling > backup.sql
cat backup.sql | docker compose -f docker/docker-compose.yml exec -T db psql -U postgres autolabeling
```

| 服务 | 地址 |
|------|------|
| 前端 | http://localhost |
| 后端 | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |

## 项目结构

完整目录树：**[docs/STRUCTURE_ZH.md](docs/STRUCTURE_ZH.md)**

## 功能

### Grounded-SAM 检测与分割

输入开放词汇类别（如 `猫`、`红色汽车`）。Grounded-SAM 在同一标注流程中执行 GroundingDINO 文本检测与 SAM2 实例分割。

- **置信度阈值**滑块（0.0–1.0，默认 0.5）控制检测灵敏度
- **Mask 阈值**滑块（0.0–1.0，默认 0.5）控制 mask 紧致度
- 可独立开关分割：纯检测模式跳过 mask 提取，更快出结果
- Grounded-SAM 作为独立 HTTP 服务运行在 8002 端口，使用独立虚拟环境（`backend/grounded-sam2-venv/`）
- 默认使用公开的 GroundingDINO + SAM2 模型；首次下载后模型缓存在 `~/.cache/huggingface/hub/`
- 首次使用时自动启动，闲置 10 分钟后自动卸载
- SSE 实时加载状态（`starting` → `loading` → `loaded`）
- 手动卸载按钮释放 GPU 内存
- 检测记录标注模型类型，便于追踪

### 视频标注

上传视频，智能提取关键帧，挑选后批量标注。

- **三种提取方式**：场景切换、运动检测（光流法）、固定间隔
- **SSIM 去重**：自动去除相似帧
- **时间轴预览**：水平滑动浏览，点击放大
- **多选机制**：勾选帧、全选/取消全选、加载到标注队列

### 手动标注

Canvas 画框模式，查看/标注双模式切换。

- 历史类别快速填充
- Grounded-SAM 预标注打底 → 删错框 → 补漏框
- 全部 / 最优 / NMS 去重过滤，设置可保存
- 临时隐藏单框，密集检测结果检查
- 单帧重新检测

### 历史管理

- 缩略图 + 类别标签，按标签多选筛选
- 点击查看详情，支持重新检测，虚拟滚动 + 无限加载
- 单张/批量导出 **5 种格式**：YOLO、YOLO-Seg、COCO JSON、Pascal VOC XML、CreateML JSON
- 下拉菜单选格式，一键下载 zip

### YOLO 训练

- **系列**：YOLOv8 / v11 / v26（n/s/m/l/x）
- **任务类型**：目标检测（Detect）、实例分割（Segment）
- 分割训练自动使用 SAM2 polygon 标签，无 polygon 时 fallback 到 bbox
- 标签筛选 + 缩略图预览精确选择训练数据
- 虚拟滚动 + 「加载全部」按钮，大数据集无卡顿
- 训练任务支持重命名，便于辨识
- 数据集拆分预设（70/20/10、80/20、90/10、60/20/20）
- SSE 实时推送：Epoch / Loss / mAP50
- 自动 ONNX 导出，可下载 PT / ONNX / 数据集 zip

### 模型验证

- **双模型源**：训练模型或外部上传 `.pt` 文件
- **Conf / IoU 滑块**实时调节阈值
- **图片批量验证**，显示边界框和置信度
- **视频验证**（三种模式）：
  - MJPEG 实时流，支持交互式暂停
  - SSE 预测流，逐帧 JSON 事件
  - 同步批量预测，一次性返回全部结果
- 临时结果，可导出 YOLO `.txt` 标注

### 模型管理

- **惰性加载**：Grounded-SAM 首次使用自动加载，闲置超时自动卸载（默认 10 分钟）
- **闲置看门狗**：Grounded-SAM 通过 `MODEL_IDLE_TIMEOUT_SECONDS` 自动卸载
- **SSE 状态**：`GET /api/v1/model/events` 推送 Grounded-SAM 服务状态
- **手动卸载**：Grounded-SAM 服务提供卸载 API
- **GPU 内存**：CUDA `expandable_segments` / `empty_cache`
- **透明加载错误**：模型加载/推理失败会把真实后端错误返回前端，不再只显示笼统的检测失败

### 排查：卡在 `加载到 GPU`

如果 Grounded-SAM 一直停在 `starting` 或 `loading`，请查看服务日志和后端日志。常见原因包括 CUDA OOM、NVIDIA runtime 未透传、驱动/runtime 不匹配、模型文件不可访问或依赖导入失败。

Docker 环境可先检查：

```bash
docker compose -f docker/docker-compose.yml logs backend --tail=200
docker compose -f docker/docker-compose.yml exec backend python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mem:", torch.cuda.mem_get_info() if torch.cuda.is_available() else None)
PY
nvidia-smi
```

## API 概览

完整 API 文档与请求/响应示例：**[docs/API_ZH.md](docs/API_ZH.md)**

## 运行平台

当前部署目标固定为 Ubuntu x86_64 + NVIDIA CUDA + Docker。Grounded-SAM 与 YOLO 训练均在 Docker 容器内运行。

## 项目亮点

- **CUDA 全链路 GPU 加速** — Grounded-SAM 和 YOLO 训练均跑 NVIDIA GPU
- **CUDA GPU 内存管理** — `expandable_segments:True`
- **Grounded-SAM 标注** — GroundingDINO 文本检测与 SAM2 实例分割组合为一个流程
- **5 种导出格式** — YOLO、YOLO-Seg、COCO、Pascal VOC、CreateML
- **检测 & 分割训练** — SAM2 polygon 标签自动用于分割训练
- **Docker-only 部署** — 面向 Ubuntu x86_64 + NVIDIA CUDA
- **智能模型生命周期** — 惰性加载、闲置自动卸载、后台下载带进度

## 开发与校验

```bash
# 前端
cd frontend && pnpm install && pnpm run lint && pnpm run build

# 后端
cd backend && source .venv/bin/activate
PYTHONPATH=. alembic upgrade head
python -m compileall app alembic
```

## 加星历史

[![Star History Chart](https://api.star-history.com/svg?repos=JackeyLa5/GroundedSAM-AutoYOLO&type=Date)](https://star-history.com/#JackeyLa5/GroundedSAM-AutoYOLO&Date)

## License

本项目代码：[AGPL-3.0](LICENSE)。

第三方依赖协议：
- GroundingDINO / SAM2 模型 — 请遵循各自上游模型许可证
- Ultralytics YOLO — [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)（copyleft，训练/部署可能触发开源义务）

---

如果这个项目对你有帮助，欢迎点个 ⭐ [Star](https://github.com/JackeyLa5/GroundedSAM-AutoYOLO)。
