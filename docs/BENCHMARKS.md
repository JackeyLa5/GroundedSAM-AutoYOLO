# Grounded-SAM Benchmarking

AutoYOLO uses Grounded-SAM for automatic annotation. The service combines GroundingDINO text detection with SAM2 instance segmentation.

## What to Measure

Record results separately for:

| Metric | Description |
|---|---|
| Cold start | Time from the first annotation request until the Grounded-SAM service is ready and returns a result |
| Detection only | Per-image latency with instance-mask extraction disabled |
| Detection + segmentation | Per-image latency with masks enabled |
| Peak accelerator memory | Maximum CUDA VRAM or Apple unified-memory use during the run |

Use representative images and record the image resolution, prompt, confidence threshold, mask threshold, hardware, operating system, and model revisions with every result.

## Default Configuration

- Grounding model: `IDEA-Research/grounding-dino-tiny`
- Segmentation model: `facebook/sam2.1-hiera-base-plus`
- Grounded-SAM service port: `8002`
- Detection confidence threshold: `0.5`
- Mask threshold: `0.5`

## Notes

- Model weights are cached by Hugging Face after the first download.
- The service starts lazily on the first annotation request and unloads after `MODEL_IDLE_TIMEOUT_SECONDS` of inactivity.
- Prompt specificity, source-image resolution, accelerator memory, and mask extraction materially affect latency.
- Re-run benchmarks after changing model revisions, accelerator software, image preprocessing, or Grounded-SAM service settings.
