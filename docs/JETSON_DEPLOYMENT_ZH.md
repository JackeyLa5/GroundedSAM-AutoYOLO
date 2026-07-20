# Jetson 机器人部署指南

本项目部署目标已从 Ubuntu x86_64 切换为 **NVIDIA Jetson Orin（aarch64，JetPack 5.x / L4T 35.x / CUDA 11.4）**。本文档记录该切换涉及的改动，以及在**每台新机器人**上从零开始部署时需要走的完整流程和常见坑。

## 一、仓库里改了什么（背景，无需重复操作）

| 文件 | 改动 | 原因 |
|---|---|---|
| `docker/backend.Dockerfile` | 基础镜像 `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` → `dustynv/pytorch:2.1-r35.4.1`，再安装锁定的 aarch64 `torchvision 0.16.2`；`TORCH_ARCH` `8.9` → `8.7` | 已验证该基础镜像提供 CUDA 11.4 的 PyTorch 2.1。`dustynv/torchvision:r35.4.1` 实际是 torch 2.0 / torchvision 0.15.1，无法满足 Grounded-SAM-2；所有后续 pip 安装均使用约束文件，避免被 PyPI 的 CPU-only torch 替换。8.7 是 Orin 系列的 GPU 计算能力 |
| `docker/backend.Dockerfile` | 新增 `RUN ln -sf "$(command -v python3)" /usr/bin/python` | `dustynv/pytorch` 基础镜像不像旧的 conda 版 `pytorch/pytorch` 镜像那样保证 `python` 命令一定指向装好 pip/torch 的解释器（实测报 `/usr/bin/python: No module named pip`） |
| `docker/backend.Dockerfile` | apt 包列表新增 `python3-pip`、`patch` | `python3-pip`：保证 `python -m pip` 可用；`patch`：`lmdb` 源码包安装时需要用 `patch` 给内置 liblmdb 打补丁，缺失会导致构建失败 |
| `docker/docker-compose.yml` | backend 新增 `group_add: ${JETSON_GPU_GID:-44}` | backend 以非 root 的 `appuser` 运行。Jetson GPU 设备节点通常属于宿主机 `video` 组（GID 44）；未加入该组会使 CUDA 初始化报 `NvRmMemInitNvmap failed with Permission denied`。 |
| `docker/backend.Dockerfile` | `git clone` + 可编辑安装 sam2/grounding_dino 之后新增 `chmod -R a+rX /opt/Grounded-SAM-2`，并将该目录加入 `PYTHONPATH` | `sam2`/`grounding_dino` 是以 root 身份用 `pip install -e .` 装的可编辑安装，实际代码在 `/opt/Grounded-SAM-2` 目录。仅设置 `/app` 为 `PYTHONPATH` 时，非 root 的预热/运行进程可能无法解析 `sam2`；显式加入源码根目录可保证导入路径可用。 |
| `docker/docker-compose.yml` | `environment` 新增 `GROUNDED_SAM2_SAM2_MODEL_ID`、`GROUNDED_SAM2_SAM2_CHECKPOINT_PATH`；`volumes` 新增 `/home/galbot/sam2/checkpoints:/models/sam2:ro` | 让 SAM2 走本地 checkpoint（`sam2.1_hiera_small.pt`），不用每次都联网从 Hugging Face 下载；改成 `hiera-small` 是因为要跟本地这份 checkpoint 的模型结构对上，见下方"本地模型文件"一节 |
| `backend/requirements.txt` / `requirements-grounded-sam.txt` | `torch>=2.6.0` → `>=2.1.0` | JetPack 5.x 最高只到 torch 2.1.0（见上） |
| `backend/requirements.txt` / `requirements-grounded-sam.txt` | `transformers==4.57.1` → `==4.46.3` | 见下方"Python 3.8 兼容性"一节 |
| `backend/requirements.txt` | `uvicorn[standard]>=0.34.0` → `>=0.30.0`；`Pillow>=11.0.0` → `>=10.0.0`；`numpy>=2.0.0` → `>=1.24.0`；新增 `eval-type-backport>=0.4.0` | 前三项为 Python 3.8 兼容性问题；最后一项让 Pydantic 能在 Python 3.8 解析 `list[str]` 这类新注解 |
| `backend/app/models/{detection,train,video}.py` | SQLAlchemy ORM 字段由 `Mapped[list[T]]` / `Mapped[T \| None]` 改为 `Mapped[List[T]]` / `Mapped[Optional[T]]` | SQLAlchemy 会在 Python 3.8 运行时解析 ORM 注解；内置泛型与 `\|` 联合类型在该版本不可用，会导致后端启动失败 |
| `backend/app/services/**/*.py` | 对仍在使用 `list[...]`、`dict[...]` 或 `T \| None` 的模块启用 `from __future__ import annotations` | 避免 Python 3.8 在导入普通函数时立即计算 Python 3.9+ 风格的类型注解 |
| `backend/alembic/versions/*.py` | 对迁移脚本启用 `from __future__ import annotations` | Alembic 会执行迁移模块的 `str \| None` 等版本元数据注解；Python 3.8 否则会在数据库初始化前失败 |
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

其余包（`fastapi`、`sqlalchemy`、`psycopg2-binary`、`pydantic-settings`、`opencv-python-headless`、`python-multipart`、`huggingface_hub`、`ultralytics`、`alembic`、`lmdb`、`peft`）经测试在 Python 3.8 下均能正常解析安装。注意 `pydantic-settings` 虽可安装，但项目的 `Settings` 使用了 `list[str]` 注解；Python 3.8 运行时须额外安装 `eval-type-backport`，否则会在导入配置时抛出 `Unable to evaluate type annotation 'list[str]'`。

### decord（架构问题，不是 Python 版本问题）

`decord>=0.6.0; sys_platform == 'linux'` 在 aarch64 上直接报 `Could not find a version that satisfies the requirement decord>=0.6.0 (from versions: none)` —— 这个包在 PyPI 上只发布了 x86_64 的预编译 wheel，没有 aarch64 wheel，也没有 sdist，无法在 Jetson 上安装。

排查后发现这行依赖其实是**多余的**：`decord` 本身没有被业务代码调用（视频关键帧抽取用的是 `ffmpeg` + `opencv`，见 `backend/app/services/video_service.py`），加它只是因为 `transformers` 内部某些代码路径会顺手 `import decord`。仓库里已经有 `backend/decord_stub.py`，会在构建后期被复制覆盖到 `site-packages/decord.py`（见 `docker/backend.Dockerfile` 中 `COPY backend/decord_stub.py` 那一步），也就是说不管 pip 有没有装真正的 `decord`，最终生效的都是这个 stub。

**处理方式**：直接从 `backend/requirements.txt` 删掉 `decord` 这一行，x86 和 aarch64 行为都不受影响（原来 pip 装的真实 decord 从未真正生效过）。

## 三、本地模型文件（跳过联网下载）

机器人网络不稳定时，可以让 SAM2 直接从本地 checkpoint 加载，不用每次都联网找 Hugging Face 下载。

### SAM2（已支持）

`backend/grounded_sam2_server.py` 的 `_build_sam2_model()` 已经内置了这个逻辑：设置了 `GROUNDED_SAM2_SAM2_CHECKPOINT_PATH` 就走本地 `build_sam2()`，没设置就走联网的 `build_sam2_hf()`。

**关键约束**：`GROUNDED_SAM2_SAM2_MODEL_ID`（决定模型结构/config）必须和本地 checkpoint 文件的实际结构一致——比如本地放的是 `sam2.1_hiera_small.pt`，`MODEL_ID` 就必须是 `facebook/sam2.1-hiera-small`，不能还留着默认的 `facebook/sam2.1-hiera-base-plus`，否则加载 state dict 时会因为网络结构对不上而报错。

`docker/docker-compose.yml` 里已经接好（按当前机器人 `/home/galbot/sam2/checkpoints/sam2.1_hiera_small.pt` 这个路径配置的，其他机器人如果路径不同要相应改）：

```yaml
environment:
  GROUNDED_SAM2_SAM2_MODEL_ID: facebook/sam2.1-hiera-small
  GROUNDED_SAM2_SAM2_CHECKPOINT_PATH: /models/sam2/sam2.1_hiera_small.pt
volumes:
  - /home/galbot/sam2/checkpoints:/models/sam2:ro
```

⚠️ 注意：`docker/backend.Dockerfile` 里 `PRELOAD_MODELS=1` 那一步（构建期间的模型预热）**没有**改成走本地 checkpoint，仍然是硬编码调用 `build_sam2_hf()` 联网下载默认模型（见该 Dockerfile 的最后一个 `RUN` 步骤）。这是刻意的：Dockerfile 要保持对没有本地 checkpoint 的机器人也通用，预热这一步多下载一次不影响正确性，只是稍微浪费构建时间/带宽。真正生效的是运行时 `grounded_sam2_server.py` 里的本地 checkpoint 逻辑。

### GroundingDINO（暂未接，继续联网下载）

代码里目前**没有**类似 SAM2 那种"本地 checkpoint 路径"开关，`AutoProcessor.from_pretrained()` / `AutoModelForZeroShotObjectDetection.from_pretrained()` 固定走 Hugging Face（或其本地缓存）。当前先保持联网下载，如果以后要接本地文件，需要准备一整套 HF 仓库文件（不是单个权重文件）：

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`（或 `pytorch_model.bin`）
- `tokenizer_config.json`、`vocab.txt`、`special_tokens_map.json`（如有 `tokenizer.json` 一并带上）

拿法：找一台能联网的机器跑 `huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir ./grounding-dino-tiny`，把整个文件夹拷到机器人上，再把 `GROUNDING_MODEL_ID` 指向这个本地目录路径即可（用法和 SAM2 的本地路径思路一致，但要接的是一整个目录而不是一个文件，目前代码还没有为它单独接类似 `GROUNDED_SAM2_SAM2_CHECKPOINT_PATH` 的开关，直接把 `GROUNDING_MODEL_ID` 设成本地目录路径即可，`from_pretrained` 原生支持）。

## 四、未解决问题

- **`ModuleNotFoundError: No module named 'sam2'`**：模型预热以 `appuser` 身份运行，而 SAM2 的可编辑安装实际引用 `/opt/Grounded-SAM-2`。Dockerfile 和 Compose 均应将该目录保留在 `PYTHONPATH`（`/app:/opt/Grounded-SAM-2`）；Dockerfile 也会在安装后立即验证 `import sam2`。
- **本地开发机 ↔ 机器人代码同步方式待定**：目前改动都在这台 x86 开发机的本地 git 仓库里，还没推送到 GitHub（`git status` 显示领先 origin/main 若干个提交），机器人是怎么拿到这些改动的还没确认（git pull？rsync？scp？）。在这个问题解决之前，每次改完文件都要先确认机器人上跑的是不是最新版本，避免像上面这次一样，重新构建后"看起来一样的报错"其实是因为改动根本没同步过去。

## 五、新机器人首次部署流程

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

#### Docker Hub 基础镜像拉取长时间无进度

执行 `docker compose ... up -d --build` 时，前端会拉取 `node:22-alpine`。如果日志中的某一层下载字节数（例如 `28.31MB / 52.67MB`）连续 5–10 分钟没有变化，通常是 Docker Hub 网络拉取阻塞，不是 Dockerfile 编译或前端构建本身出错。

可以按 `Ctrl+C` 安全中断本次构建（不会删除已有容器或命名卷），单独拉取基础镜像后再重试：

```bash
docker pull node:22-alpine
docker compose -f docker/docker-compose.yml up -d --build db backend frontend
```

若单独拉取仍然没有进度，检查 Docker Hub 网络连通性与可用磁盘空间：

```bash
df -h
docker system df
```

### 6. 从 PC 访问

```bash
hostname -I   # 查看机器人 IP
```

`docker-compose.yml` 里 `ports: ["80:80"]` 默认监听所有网卡，PC 和机器人在同一网络下，浏览器直接访问 `http://<机器人IP>/` 即可，无需为每台机器人单独配置或重新构建前端（后端地址走的是 nginx 同源反代，不依赖具体 IP）。

## 六、其他排查提示

- **是否需要在机器人上手动 clone sam2 / GroundingDINO 仓库？不需要。** `docker/backend.Dockerfile` 构建镜像时会自动 `git clone` `Baijing0817/Grounded-SAM-2`（见一、表格第 13 行）到 `/opt/Grounded-SAM-2`，并对其根目录和 `grounding_dino/` 子目录分别执行 `pip install -e .`，一次性提供 `sam2` 和 `grounding_dino` 两个包（editable 安装）。`backend/requirements.txt` 里的 `sam2>=1.0.0` 那一行在 Jetson 构建时会被过滤掉（不从 PyPI 装），只是给非 Jetson 环境占位用的，真正生效的是 git clone 出来的这份代码。机器人上唯一要手动准备的是**权重文件**而不是仓库代码：SAM2 需要把 `sam2.1_hiera_small.pt` 放到 `/home/galbot/sam2/checkpoints/`（见"三、本地模型文件"），GroundingDINO 目前继续联网从 Hugging Face 下载（模型 ID `IDEA-Research/grounding-dino-tiny`），暂无需手动下载文件。
- `shm_size: "8gb"`（`docker/docker-compose.yml`）：Jetson 是 CPU/GPU 统一内存架构，如果是 Orin Nano（8GB 内存）这种低内存型号，可能需要调低这个值，否则容器启动可能受影响。
- 以后凡是要在**已经 source 过 ROS 环境**的终端里执行 `curl` / `wget` / `apt` / `pip` 这类依赖系统库或联网的命令，遇到奇怪的 "no version information" 警告或 SSL 证书错误，优先怀疑 `LD_LIBRARY_PATH` 被 `/opt/ros/...` 或机器人 SDK 路径污染，用 `LD_LIBRARY_PATH= <command>` 临时清空排查。
