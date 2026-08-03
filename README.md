# Episodic Memory Pipeline

Benchmarking pipeline to assess Vision Language Models' (VLMs) episodic memory capabilities in long-form egocentric dashcam video.

## Setup

Create a `.env` file at the project root with your API key:

```bash
# .env
OPENCODE_API_KEY=sk-or-v1-...
```

## Structure

| Script | Purpose |
|--------|---------|
| `pipeline/sparse_event_main.py` | `SparseEventPipeline` class; creates the config loader, frame parser, and AI parser |
| `pipeline/temporal_main.py` | Reserved for the temporal pipeline class |
| `pipeline/ConfigLoader.py` | Shared JSON configuration loader for pipeline entry points |
| `pipeline/frame_parser.py` | Splits a folder of numerically named frame images into VLM-sized sections |
| `pipeline/AIParser.py` | Local vLLM wrapper; `call_llm` handles text and `call_vlm` handles all frames in a folder |
| `pipeline/prompts/*.txt` | VLM prompt files, one per benchmark task |
| `configs/*.json` | Config files for each pipeline run |
| `main.py` | Coworker prototype — self-hosted vLLM, reference only |

The pipeline classes are currently created by a future top-level runner. The
`SparseEventPipeline` class can be run programmatically with a config path.

```python
from pipeline.sparse_event_main import SparseEventPipeline

SparseEventPipeline("configs/sparse_events.json").run()
```

Each run:
1. Splits frames into sections (`frame_parser`)
2. Queries the VLM for every section (`AIParser`)
3. Writes per-section JSON + an aggregated `all_results.json`

## Local VLM Calls

`AIParser.call_vlm(prompt, folder_path)` accepts a folder containing image frames.
It loads all `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files, sorts them by
their numeric filename, and sends them together to the local vLLM model as a
multi-image input. `pipeline/sparse_event_main.py` loads the JSON config and constructs the
parser; direct callers can construct `AIParser` with the model settings they
need.

```python
from pipeline.AIParser import AIParser

parser = AIParser(model="Qwen/Qwen2-VL-7B-Instruct")
response = parser.call_vlm(
    "Describe the noteworthy events in these frames.",
    "./sections/section_0000",
)
```

## Config files

All parameters live in JSON. Create your own or edit the examples in `configs/`.

```jsonc
{
    "task": 1,                    // 1 = sparse events, 2 = temporal/narrative
    "frames_dir": "./frames",     // Directory of frame images
    "sections_dir": "./sections", // Intermediate section output
    "output": "./results",        // Final VLM results
    "frames_per_section": 100,    // Frames per section (default 100)
    "step": 2,                    // Keep every Nth frame (default 1)
    "model": "Qwen/Qwen2-VL-7B-Instruct", // Local vLLM model
    "temperature": 0.2,
    "max_tokens": 100,
    "enforce_eager": true,
    "dtype": "half",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "move": false
}
```

| `task` | Prompt used | Purpose |
|--------|-------------|---------|
| 1 | `prompts/sparse_event_prompt.txt` | Sparse event localisation (find rare/noteworthy events) |
| 2 | `prompts/temporal.txt` | Full scene description (foundation for temporal/episodic memory) |

## Output

```text
<output>/
    all_results.json                    # Aggregated, single-file view
    section_0000_output/result.json     # Per-section artifact
    section_0001_output/result.json
    ...
```

## Full Documentation

See [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md) for a detailed breakdown of the entire codebase.
