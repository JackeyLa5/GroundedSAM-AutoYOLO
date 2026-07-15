# AutoYOLO 项目架构设计图

AutoYOLO 是一个用于 Grounded-SAM 自动标注、人工校对和 YOLO 训练的全栈平台。自动标注链路统一为 GroundingDINO 文本检测加 SAM2 实例分割。

```mermaid
graph TD
    User((用户)) --> Frontend["前端<br/>Vite + React + TypeScript"]
    Frontend --> API["FastAPI API"]
    API --> Detection["检测编排服务"]
    Detection --> GroundedSAM["Grounded-SAM 服务<br/>GroundingDINO + SAM2<br/>grounded_sam2_server.py:8002"]
    Detection --> Storage[("PostgreSQL + 文件系统")]
    API --> YOLO["YOLO 训练与验证"]
    YOLO --> Storage
    GroundedSAM --> Cache[("Hugging Face 模型缓存")]
```

## 架构说明

### 前端

React 单页应用负责图片、视频和关键帧上传，Grounded-SAM 参数配置，Canvas 标注校对，数据集导出以及 YOLO 训练和验证。

### 后端

FastAPI 路由接收请求，检测服务编排 Grounded-SAM 调用并持久化结果；Repository 和 SQLAlchemy 负责结构化数据访问，文件系统保存媒体、标注和训练输出。

### Grounded-SAM 服务

`backend/grounded_sam2_server.py` 作为独立 HTTP 服务运行。GroundingDINO 根据开放词汇文本提示生成检测框，SAM2 根据检测实例生成 mask。关闭分割时返回检测框，不会切换到另一套模型。

### YOLO 闭环

人工修正后的 bbox 和 polygon 标注可导出为多种格式，并用于 YOLOv8、YOLOv11 或 YOLOv26 的检测/分割训练和后续验证。

## 核心业务流程

```mermaid
sequenceDiagram
    actor User as 用户
    participant UI as 前端
    participant API as FastAPI
    participant GS as Grounded-SAM
    participant DB as 数据库/文件系统
    participant YOLO as YOLO 训练与验证

    User->>UI: 上传图片、视频或关键帧
    UI->>API: 输入开放词汇提示并发起标注
    API->>GS: GroundingDINO 检测
    GS-->>API: 返回 bbox
    opt 启用分割
        API->>GS: SAM2 实例分割
        GS-->>API: 返回 mask/polygon
    end
    API-->>UI: 渲染 Grounded-SAM 标注结果
    User->>UI: 增删改标注并保存
    UI->>API: 导出数据集或创建训练任务
    API->>DB: 持久化标注和训练数据
    API->>YOLO: 训练或验证
    YOLO-->>UI: 返回进度与预测结果
```
