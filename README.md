# Episodic Memory Pipeline

Benchmarking pipeline to assess Vision Language Models' (VLMs) episodic memory capabilities in long-form egocentric dashcam video.

## Structure

| Script | Purpose |
|--------|---------|
| `main.py` | Full-video chunked description pipeline using vLLM (Qwen2-VL + DeepSeek-V4) |
| `sparse_event_pipeline/frame_parser.py` | Split an existing folder of numerically named frame images into VLM-sized sections |
| `sparse_event_pipeline/AIParser.py` | Concurrent subsection parser using the OpenCode Go VLM API |
| `sparse_event_pipeline/run.py` | CLI wrapper for the subsection VLM parser |

## Quick Start

```bash
# 1. Split an existing frame folder into sections.
#    Frame filenames must have numeric stems, such as 1733343593917869.png.
python sparse_event_pipeline/frame_parser.py path/to/frames --output path/to/sections --frames-per-section 100 --step 2

# 2. Query every generated subsection concurrently.
export OPENCODE_API_KEY="your-api-key"
python sparse_event_pipeline/run.py path/to/sections \
  --output path/to/results \
  --model qwen3.7-plus \
  --max-concurrent 3
```

`--step 1` keeps every frame. A value of `2` keeps every second frame,
which reduces the number of images sent to the VLM. Frames are sorted by their
numeric filename before sampling and sectioning. By default, frames are copied;
pass `--move` to move them instead. Each subsection result is written to a
directory such as `section_0000_output/result.json` under the output directory.
 
## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the entire codebase.
