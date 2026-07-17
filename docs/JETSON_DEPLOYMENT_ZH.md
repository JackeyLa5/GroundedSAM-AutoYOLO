# Jetson 机器人部署指南

本项目部署目标已从 Ubuntu x86_64 切换为 **NVIDIA Jetson Orin（aarch64，JetPack 5.x / L4T 35.x / CUDA 11.4）**。本文档记录该切换涉及的改动，以及在**每台新机器人**上从零开始部署时需要走的完整流程和常见坑。

## 一、仓库里改了什么（背景，无需重复操作）

| 文件 | 改动 | 原因 |
|---|---|---|
| `docker/backend.Dockerfile` | 基础镜像 `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` → `dustynv/pytorch:2.1-r35.4.1`；`TORCH_ARCH` `8.9` → `8.7` | JetPack 5.x（CUDA 11.4，aarch64）没有 torch 2.6 的官方 wheel，最高只到 2.1.0；8.7 是 Orin 系列的 GPU 计算能力 |
| `docker/backend.Dockerfile` | 新增 `RUN ln -sf "$(command -v python3)" /usr/bin/python` | `dustynv/pytorch` 基础镜像不像旧的 conda 版 `pytorch/pytorch` 镜像那样保证 `python` 命令一定指向装好 pip/torch 的解释器（实测报 `/usr/bin/python: No module named pip`） |
| `docker/backend.Dockerfile` | apt 包列表新增 `python3-pip`、`patch` | `python3-pip`：保证 `python -m pip` 可用；`patch`：`lmdb` 源码包安装时需要用 `patch` 给内置 liblmdb 打补丁，缺失会导致构建失败 |
| `backend/requirements.txt` / `requirements-grounded-sam.txt` | `torch>=2.6.0` → `>=2.1.0` | JetPack 5.x 最高只到 torch 2.1.0（见上） |
| `backend/requirements.txt` / `requirements-grounded-sam.txt` | `transformers==4.57.1` → `==4.46.3` | 见下方"Python 3.8 兼容性"一节 |
| `backend/requirements.txt` | `uvicorn[standard]>=0.34.0` → `>=0.30.0`；`Pillow>=11.0.0` → `>=10.0.0`；`numpy>=2.0.0` → `>=1.24.0` | 同上，均为 Python 3.8 兼容性问题 |
| `docker/docker-compose.yml` | 镜像 tag `ubuntu-x86*` → `jetson-arm64*`；GPU 挂载从 `deploy.resources.reservations.devices`（driver: nvidia）改为 `runtime: nvidia` | Jetson 走 `nvidia-container-runtime`，不是 x86 独立显卡那套 CDI/device-reservation 语法 |
| `CLAUDE.md` | 部署目标说明同步更新为 Jetson Orin / JetPack 5.x | 保持文档与实际一致 |

前端（`frontend/`）无需改动：`API_BASE` 默认走相对路径 `/api/v1`，由 `docker/nginx.conf` 反向代理到后端容器，跟机器人 IP 无关，不需要为每台机器人单独构建。

## 二、Python 3.8 兼容性（`requirements.txt` 版本下限踩坑记录）

`dustynv/pytorch:2.1-r35.4.1` 自带 **Python 3.8**（JetPack 5.x / Ubuntu 20.04 默认版本）。原 `requirements.txt` 里不少包的版本下限是按 Python 3.9+ 环境写的，这些包官方从某个版本起就不再发布支持 3.8 的构建，在机器人上会直接报 `Could not find a version that satisfies the requirement`。

排查方法：不用每次跑机器人上又慢又要联网的真实构建去试错，而是本地起一个 `python:3.8-slim` 容器（跟机器人 Python 版本一致，架构不影响这类"包声明的最低 Python 版本"限制）反复跑 `pip install --dry-run` 定位问题包，确认后再统一改仓库文件。

| 包 | 原下限 | 改为 | 原因 |
|---|---|---|---|
| `transformers` | `==4.57.1` | `==4.46.3` | 4.57.1 需要 Python≥3.9；同时 4.52.4（第一次改的版本）也需要 Python≥3.9。4.46.3 是 Python 3.8 下能装到的最新版本，仍支持 Grounding DINO（4.40.0 引入），且早于 `torch.compiler.is_compiling` 那个 bug 引入的版本（4.53.0） |
| `uvicorn[standard]` | `>=0.34.0` | `>=0.30.0` | 0.34.0 起要求 Python≥3.9，3.8 下最高能装到 0.33.0 |
| `Pillow` | `>=11.0.0` | `>=10.0.0` | 11.0.0 起要求 Python≥3.9，3.8 下最高能装到 10.4.0 |
| `numpy` | `>=2.0.0` | `>=1.24.0` | 2.0.0 起要求 Python≥3.9，3.8 下最高能装到 1.24.4 |

以上下限调低**不影响**未来若在更新的 Python 环境（比如本地 x86 开发机）下 `pip install`：pip 总是优先选满足条件的最新版本，只有在 Python 3.8 这种环境下才会被迫落到旧版本。

其余包（`fastapi`、`sqlalchemy`、`psycopg2-binary`、`pydantic-settings`、`opencv-python-headless`、`python-multipart`、`huggingface_hub`、`ultralytics`、`alembic`、`lmdb`、`peft`、`decord`）经测试在 Python 3.8 下均能正常解析安装，未做改动。

## 二、新机器人首次部署流程

### 1. 确认系统环境

```bash
cat /etc/nv_tegra_release        # 确认是 JetPack 5.x / L4T 35.x
docker -v                        # 确认 Docker 本体已装（JetPack SDK Manager 通常已带）
docker compose version           # 大概率会报错，见下一步
```

### 2. 安装 Docker Compose V2 插件

Jetson 出厂/SDK Manager 装的 Docker 往往只有引擎本体，没有 `docker compose` 子命令（`docker-compose` 独立二进制也不一定有），需要单独装插件：

```bash
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
```

如果 apt 提示找不到 `docker-compose-plugin` 这个包，说明当前 apt 源里没有 Docker 官方仓库，需要先加上：

```bash
sudo rm -f /usr/share/keyrings/docker-archive-keyring.gpg

# ⚠️ 如果这台机器人的终端 source 过 ROS（LD_LIBRARY_PATH 里有 /opt/ros/.../lib、
# /data/galbot/lib 之类的路径），curl 会被这些自带的旧版 libcurl.so 顶替，
# 导致 SSL 证书校验失败（报 "no version information" / 证书错误）。
# 用 LD_LIBRARY_PATH= 临时清空该变量即可，只影响这一条命令，不影响机器人程序运行。
LD_LIBRARY_PATH= curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker.gpg
sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg /tmp/docker.gpg
ls -la /usr/share/keyrings/docker-archive-keyring.gpg   # 确认文件不是 0 字节

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

装插件前**先不加 `-y` 跑一遍**，确认它只装 `docker-compose-plugin`（可能附带 `docker-buildx-plugin`），**没有**牵连升级/降级 `docker-ce`、`containerd` 等核心包（那些是 JetPack 已经配好 NVIDIA runtime 的部分，被 apt 顺带升级有风险）：

```bash
sudo apt-get install docker-compose-plugin   # 看清单，确认没有 docker-ce/containerd 再输入 y
docker compose version                        # 验证装好
```

> 常见无害警告：`APT had planned for dpkg to do more than it reported back ... nvidia-l4t-bootloader/nvidia-l4t-kernel` —— 这只是 dpkg 触发器计数提示，只要这两个包不在实际安装/升级列表里就可以忽略。

### 3. 确认 NVIDIA container runtime 已注册

```bash
docker info | grep -i runtime   # 应该能看到 "nvidia"
```

没有的话检查 `/etc/docker/daemon.json` 是否有 `nvidia` runtime 配置（一般 JetPack SDK Manager 已经配好）。

### 4. 构建前端静态文件

`docker-compose.yml` 会用 host 上的 `frontend/dist` 覆盖镜像内构建结果（bind mount），所以启动前需要保证这个目录存在且是最新构建：

```bash
cd frontend
pnpm install
pnpm run build
cd ..
```

（也可以在别的机器上构建好 `frontend/dist` 再拷贝到机器人上，静态产物和架构无关。）

### 5. 构建并启动

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml logs -f backend
```

首次构建注意：

- `dustynv/pytorch:2.1-r35.4.1` 镜像体积较大，下载视网络情况可能较慢。
- GroundingDINO / SAM2 的 CUDA 扩展会在构建时用 `ninja` 现场编译，Orin 的 CPU 编译速度明显慢于 x86 开发机，构建卡住不动是正常现象，不代表失败。
- `PRELOAD_MODELS=1` 会在构建期间从 Hugging Face 下载 `grounding-dino-tiny` 和 `sam2.1-hiera-base-plus`，如遇限流/需要私有模型，设置 `HF_TOKEN` 环境变量。

### 6. 从 PC 访问

```bash
hostname -I   # 查看机器人 IP
```

`docker-compose.yml` 里 `ports: ["80:80"]` 默认监听所有网卡，PC 和机器人在同一网络下，浏览器直接访问 `http://<机器人IP>/` 即可，无需为每台机器人单独配置或重新构建前端（后端地址走的是 nginx 同源反代，不依赖具体 IP）。

## 三、其他排查提示

- `shm_size: "8gb"`（`docker/docker-compose.yml`）：Jetson 是 CPU/GPU 统一内存架构，如果是 Orin Nano（8GB 内存）这种低内存型号，可能需要调低这个值，否则容器启动可能受影响。
- 以后凡是要在**已经 source 过 ROS 环境**的终端里执行 `curl` / `wget` / `apt` / `pip` 这类依赖系统库或联网的命令，遇到奇怪的 "no version information" 警告或 SSL 证书错误，优先怀疑 `LD_LIBRARY_PATH` 被 `/opt/ros/...` 或机器人 SDK 路径污染，用 `LD_LIBRARY_PATH= <command>` 临时清空排查。
