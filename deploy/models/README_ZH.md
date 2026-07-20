# 内置模型放置目录

这个目录仅用于制作**模型内置 backend 镜像**。模型权重被 `.gitignore`
忽略，不会提交到 Git。

在构建离线镜像前，目录必须是：

```text
deploy/models/
├── sam2/
│   └── sam2.1_hiera_small.pt
└── grounding-dino-tiny/
    ├── config.json
    ├── preprocessor_config.json
    ├── model.safetensors                 # 或 pytorch_model.bin
    └── tokenizer 等其余 Hugging Face 文件
```

SAM2 来源是当前已验证机器上的：

```text
/home/galbot/sam2/checkpoints/sam2.1_hiera_small.pt
```

GroundingDINO 必须是完整模型目录，不能只复制 `model.safetensors`。建议通过：

```bash
huggingface-cli download IDEA-Research/grounding-dino-tiny \
  --local-dir deploy/models/grounding-dino-tiny
```

离线构建会校验 SAM2 文件、GroundingDINO 的配置、预处理器配置和权重文件。
