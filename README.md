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
| `sparse_event_pipeline/run.py` | Single entry point — runs the full pipeline (frame splitting + VLM query) from one JSON config |
| `sparse_event_pipeline/frame_parser.py` | Splits a folder of numerically named frame images into VLM-sized sections |
| `sparse_event_pipeline/AIParser.py` | Library — concurrent subsection parser via OpenCode Go VLM API; `call_llm` and `call_vlm` for downstream modules |
| `sparse_event_pipeline/prompts/*.txt` | VLM prompt files, one per benchmark task |
| `configs/*.json` | Config files for each pipeline run |
| `main.py` | Coworker prototype — self-hosted vLLM, reference only |

## Quick Start

```bash
# Sparse event detection (task 1)
python sparse_event_pipeline/run.py --config configs/sparse_events.json

# Full scene descriptions (task 2)
python sparse_event_pipeline/run.py --config configs/narrative_pass.json
```

Each run:
1. Splits frames into sections (`frame_parser`)
2. Queries the VLM for every section (`AIParser`)
3. Writes per-section JSON + an aggregated `all_results.json`

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
    "model": "qwen3.7-plus",      // VLM model on the gateway
    "max_concurrent": 3           // Concurrent API calls (default 3)
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
