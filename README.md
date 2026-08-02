# Episodic Memory Pipeline

Benchmarking pipeline to assess Vision Language Models' (VLMs) episodic memory capabilities in long-form egocentric dashcam video.

## Structure

| Script | Purpose |
|--------|---------|
| `main.py` | Coworker prototype — full-video chunked description with change-detection QA drafts (self-hosted vLLM, reference only) |
| `sparse_event_pipeline/frame_parser.py` | Split an existing folder of numerically named frame images into VLM-sized sections |
| `sparse_event_pipeline/AIParser.py` | Concurrent subsection parser via OpenCode Go VLM API; reusable `call_llm`/`call_vlm` for downstream modules |
| `sparse_event_pipeline/run.py` | CLI wrapper for the subsection VLM parser |
| `sparse_event_pipeline/prompts/describe_scene.txt` | Full-narrative VLM prompt for the temporal pipeline |

## Quick Start

```bash
# 1. Split an existing frame folder into sections.
#    Frame filenames must have numeric stems, such as 1733343593917869.png.
python sparse_event_pipeline/frame_parser.py path/to/frames \
  --output path/to/sections \
  --frames-per-section 100 \
  --step 2

# 2. Pass 1: sparse event detection (built-in default prompt).
export OPENCODE_API_KEY="your-api-key"
python sparse_event_pipeline/run.py path/to/sections \
  --output path/to/events \
  --model qwen3.7-plus \
  --max-concurrent 3

# 3. Pass 2: full scene descriptions for temporal QA
#    (narrative prompt file, custom output filename).
python sparse_event_pipeline/run.py path/to/sections \
  --output path/to/narratives \
  --prompt-file sparse_event_pipeline/prompts/describe_scene.txt \
  --output-name narrative.json \
  --max-concurrent 3
```

`--step 1` keeps every frame. A value of `2` keeps every second frame,
which reduces the number of images sent to the VLM. Frames are sorted by
numeric filename before sampling and sectioning. By default, frames are
copied; pass `--move` to move them instead.

Each subsection result is written to a directory such as
`section_0000_output/result.json` under the output directory. Use
`--output-name` to change the per-section filename and `--prompt-file` to
load the VLM prompt from a text file. `AIParser.py` also exposes `call_llm`
(text-only API call) and `call_vlm` (images + text) as reusable methods for
downstream pipeline stages.
 
## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the entire codebase.
