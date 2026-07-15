# Grounded-SAM 性能测试

AutoYOLO 的自动标注统一使用 Grounded-SAM。该服务将 GroundingDINO 文本检测与 SAM2 实例分割组合为一个流程。

## 建议记录的指标

| 指标 | 说明 |
|---|---|
| 冷启动 | 首次标注请求到 Grounded-SAM 服务就绪并返回结果的耗时 |
| 纯检测 | 关闭实例 mask 提取时的单图耗时 |
| 检测加分割 | 启用 mask 时的单图耗时 |
| 峰值加速器内存 | 运行期间的最大 CUDA 显存或 Apple 统一内存占用 |

每次测试应同时记录图像分辨率、提示词、置信度阈值、mask 阈值、硬件、操作系统和模型版本。

## 默认配置

- 检测模型：`IDEA-Research/grounding-dino-tiny`
- 分割模型：`facebook/sam2.1-hiera-base-plus`
- Grounded-SAM 服务端口：`8002`
- 检测置信度阈值：`0.5`
- Mask 阈值：`0.5`

## 说明

- 模型权重会在首次下载后由 Hugging Face 缓存。
- 服务会在首次标注请求时惰性启动，并在 `MODEL_IDLE_TIMEOUT_SECONDS` 指定的空闲时间后卸载。
- 提示词具体程度、原图分辨率、加速器内存和是否提取 mask 都会影响耗时。
- 更新模型版本、加速器软件、图像预处理或 Grounded-SAM 服务配置后，应重新测试。
