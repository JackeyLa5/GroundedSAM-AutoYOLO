# Jetson 离线内置模型发布

此目录用于制作**模型已经在 backend 镜像内部**的离线发布包。

发布后的新 Orin 不需要存放 SAM2 或 GroundingDINO 文件，也不需要联网下载基础镜像、Python 依赖或模型。它只需：

```bash
./install-on-orin.sh
```

## 这套配置与当前开发配置的区别

`docker/docker-compose.yml` 是当前开发/调试配置：它会挂载源码、日志以及宿主机上的 SAM2 权重。

`deploy/docker-compose.yml` 是发布运行配置：它没有 `build:`、没有源码挂载、没有宿主机模型挂载。代码和两个模型都来自已经导入的镜像。

因此不要用发布 Compose 覆盖当前的 `docker/docker-compose.yml`；两者可以同时保留。

## 一、制作机：准备模型

制作机必须是已经验证通过的 Jetson Orin，并具备与目标机器兼容的 JetPack/L4T 版本。项目根目录执行：

```bash
./deploy/prepare-models.sh
```

它会：

1. 从 `/home/galbot/sam2/checkpoints/sam2.1_hiera_small.pt` 复制 SAM2 权重；
2. 下载 `IDEA-Research/grounding-dino-tiny` 的完整目录；
3. 将它们放入 `deploy/models/`，供 Docker 构建时复制到 `/opt/models/`。

如果 SAM2 文件不在默认位置，传入实际路径：

```bash
./deploy/prepare-models.sh /实际路径/sam2.1_hiera_small.pt
```

## 二、制作机：构建并导出发布包

```bash
./deploy/export-release.sh
```

脚本会检查模型是否齐全，构建两个离线标签镜像：

```text
autoyolo-backend:jetson-arm64-cuda-offline
autoyolo-frontend:jetson-arm64-offline
```

并在 `dist/autoyolo-jetson-offline-日期时间/` 生成：

```text
docker-compose.yml
README_ZH.md
install-on-orin.sh
images/
├── autoyolo-jetson-images.tar
└── SHA256SUMS
```

`autoyolo-jetson-images.tar` 内含 backend、frontend 与 PostgreSQL 镜像；backend 内已包含 SAM2 和 GroundingDINO。

构建完成后，若要在制作机试运行发布镜像，必须先停止当前占用 80、8000、8002 端口的开发服务：

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```

验证完成后，若要恢复开发配置：

```bash
docker compose -f deploy/docker-compose.yml down
docker compose -f docker/docker-compose.yml up -d
```

## 三、新 Orin：导入并启动

### 0. 全新机器人：先确认宿主机前置环境

如果这台 Orin 是第一次部署、还没跑过任何 Docker 项目，`install-on-orin.sh` 依赖的宿主机环境（Docker、Compose V2、NVIDIA Container Runtime）不一定齐全，建议先手动确认一遍，省得脚本中途报错反复排查：

```bash
cat /etc/nv_tegra_release        # 确认是 JetPack 5.x / L4T 35.x
docker -v                        # 确认 Docker 本体已装（JetPack SDK Manager 通常已带）
docker compose version           # 大概率会报错，见下一步
```

**没有 `docker compose` 子命令**：Jetson 出厂/SDK Manager 装的 Docker 往往只有引擎本体，没有 Compose V2 插件，需要单独装：

```bash
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
```

如果 apt 提示找不到 `docker-compose-plugin` 这个包，说明当前 apt 源里没有 Docker 官方仓库，需要先加上：

```bash
sudo rm -f /usr/share/keyrings/docker-archive-keyring.gpg

# ⚠️ 先确认 curl 已安装且能正常执行：`curl: command not found` 会让下面这条
# 下载命令静默失败（不下载 gpg key），进而导致 keyring 文件是空的，
# apt-get update 报 NO_PUBKEY，最终还是找不到 docker-compose-plugin。
sudo apt-get install -y curl

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

**确认 NVIDIA Container Runtime 已注册**：

```bash
docker info | grep -i runtime   # 应该能看到 "nvidia"
```

没有的话检查 `/etc/docker/daemon.json` 是否有 `nvidia` runtime 配置（一般 JetPack SDK Manager 已经配好）。

> 报 `permission denied while trying to connect to the Docker daemon socket`：当前用户不在 `docker` 组，跟 nvidia runtime 无关。执行：
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker   # 让当前终端立即生效，无需重新登录
> ```
> **注意**：`newgrp` 只对执行它的那一个终端会话生效。如果你在另一个终端窗口（例如运行 `install-on-orin.sh` 的那个）里仍然报 nvidia runtime 未注册，先确认是不是那个窗口没有走过 `newgrp`／重新登录——`docker info` 权限被拒绝时脚本里的 `2>/dev/null` 会吞掉报错，容易被误判成"runtime 未注册"。

以上都确认无误后，再继续下面的导入步骤。

### 1. 复制并运行安装脚本

把整个 `dist/autoyolo-jetson-offline-日期时间` 文件夹复制到新 Orin，然后执行：

```bash
cd autoyolo-jetson-offline-日期时间
./install-on-orin.sh
```

脚本会检查 arm64、Docker、Compose V2、`/dev/nvmap`、NVIDIA Container Runtime 和镜像校验值；通过后自动写入该机器实际的 `JETSON_GPU_GID`、导入镜像并启动服务。

容器刚启动时看到 `health: starting` 是正常现象。等待约 40 秒，再执行：

```bash
docker compose ps
```

预期 backend、db、frontend 均为 `healthy`。

### 2. 完整验收：GPU、接口和模型

验证 GPU：

```bash
docker compose exec backend \
  python -c 'import torch; print(torch.version.cuda); print(torch.cuda.is_available())'
```

预期输出 CUDA `11.4` 与 `True`。

验证后端接口：

```bash
docker compose exec backend \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/health').read().decode())"
```

预期包含：

```json
{"status":"ok","version":"0.1.0"}
```

验证两种模型确实已在 backend 镜像内：

```bash
docker compose exec backend sh -lc \
  'ls -lh /opt/models/sam2/sam2.1_hiera_small.pt && \
   ls /opt/models/grounding-dino-tiny/config.json'
```

全部通过后，从与 Orin 同一网络的电脑浏览器访问：

```text
http://<Orin-IP>/
```

## 四、安装后如何使用

### 1. 从电脑打开网页

先在新 Orin 上查看局域网 IP：

```bash
hostname -I
```

选择和你的电脑处于同一网络的那个 IP。例如 Orin 的该网卡 IP 是
`192.168.38.184`，电脑也接入这个网络后，在电脑浏览器打开：

```text
http://192.168.38.184/
```

若浏览器无法访问，先确认电脑与 Orin 是否接入同一子网；Docker 已将前端的 80 端口
发布到 Orin 的所有网卡，无需为每台机器人重新构建前端。

### 2. 网页标注流程

1. 选择“自动标注”；
2. 上传图片或视频；
3. 输入目标类别/提示词，例如 `cup`、`bottle`、`red cup`；
4. 需要精细轮廓时，勾选 Grounded-SAM 分割以生成框和 mask；
5. 点击“开始检测”；
6. 检查、修改或过滤结果；
7. 点击“保存过滤结果”；
8. 点击“导出标注”或“导出数据集”，用于后续 YOLO 训练。

首次实际检测时，GroundingDINO/SAM2 会加载到 GPU；请等待页面显示模型已加载。首次加载
CUDA 扩展时可能比后续检测慢一些。

### 3. 日常管理

所有以下命令都在发布文件夹中执行：

```bash
# 查看三个服务状态
docker compose ps

# 跟踪后端日志（排查检测失败时最有用）
docker compose logs -f backend

# 停止服务；数据库、上传图片和训练数据仍保留
docker compose down

# 再次启动
docker compose up -d
```

上传图片、标注结果、数据库和训练数据保存在新 Orin 的 Docker 数据卷中，而不是发布文件夹
本身。不要使用 `docker compose down -v`，除非明确要删除这些运行数据。

## 注意事项

- 所有目标机器必须是 arm64 Jetson，且 JetPack/L4T 与制作机兼容；不能将此镜像用于普通 x86 PC。
- 离线包不能替代 JetPack、Docker、Compose 或 NVIDIA Container Runtime；这些是 Docker 启动前的宿主机依赖。脚本会在缺失时停止并指出缺哪项。
- GroundingDINO 的 CUDA 扩展可能会在新机首次实际检测时编译一次，之后会保留在容器的运行缓存中；这会占用一些时间和 CPU，但不需要联网。
- 模型内置后，更换 SAM2 或 GroundingDINO 都需要重新运行发布脚本，得到新的镜像包。
- `docker save` 不包括数据库、上传图片和训练数据；首次交付包是干净数据。不要执行 `docker compose down -v`，除非确实要删除这些运行数据。
- `group_add` 默认使用 Jetson 常见的 GPU 设备组 GID 44。若 GPU 不可用，在新机器执行 `stat -c '%g' /dev/nvmap`，再执行 `export JETSON_GPU_GID=<结果>` 后重建 backend 容器。
