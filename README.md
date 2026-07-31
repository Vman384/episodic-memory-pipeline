# Episodic Memory Pipeline

Benchmarking pipeline to assess Vision Language Models' (VLMs) episodic memory capabilities in long-form egocentric dashcam video.

## Structure

| Script | Purpose |
|--------|---------|
| `main.py` | Full-video chunked description pipeline using vLLM (Qwen2-VL + DeepSeek-V4) |
| `sparse_event_pipeline/video_parser.py` | Build frame-range manifest for overlapping video sections (Decord, no ffmpeg) |
| `sparse_event_pipeline/AIParser.py` | Sparse event detector using opencode.ai VLM gateway (Anthropic-compatible) |

## Quick Start

```bash
# 1. Generate a section manifest
python sparse_event_pipeline/video_parser.py my_dashcam_video.mp4 --output manifest.json --duration 300 --overlap 10

# 2. Detect sparse events
python sparse_event_pipeline/AIParser.py manifest.json --output events.json --fps 1.0
```
 
## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the entire codebase.
