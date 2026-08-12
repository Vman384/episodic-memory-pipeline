import argparse
import sys
from enum import Enum
from pathlib import Path


class BenchmarkMode(Enum):
    SPARSE_EVENT = "sparse"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COUNTING = "counting"


MODE_CONFIG = {
    BenchmarkMode.SPARSE_EVENT: "configs/sparse_events.json",
    BenchmarkMode.TEMPORAL: "configs/narrative_pass.json",
    BenchmarkMode.SPATIAL: None,
    BenchmarkMode.COUNTING: None,
}


def main():
    parser = argparse.ArgumentParser(description="Episodic Memory Benchmark Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[m.value for m in BenchmarkMode],
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["extract", "timeline", "questions"],
        help="Temporal pipeline stage. By default temporal runs extract then "
        "timeline. Run questions only after a human has reviewed timeline.json.",
    )

    args = parser.parse_args()

    mode = BenchmarkMode(args.mode)

    if mode == BenchmarkMode.SPARSE_EVENT:
        from pipeline.SparseEventPipeline import SparseEventPipeline

        config_path = MODE_CONFIG[mode]
        prompt_path = Path("pipeline/prompts/sparse_event_prompt.txt")
        if not prompt_path.is_file():
            raise SystemExit(f"Prompt file not found: {prompt_path}")

        prompt = prompt_path.read_text()
        pipeline = SparseEventPipeline(config_path, prompt)
        pipeline.run()

    elif mode == BenchmarkMode.TEMPORAL:
        from pipeline.TemporalPipeline import TemporalPipeline

        prompt_files = {
            "prompt": "pipeline/prompts/temporal_vlm.txt",
            "filter_prompt": "pipeline/prompts/temporal_llm_filter.txt",
            "storyline_prompt": "pipeline/prompts/temporal_storyline.txt",
            "question_prompt": "pipeline/prompts/temporal_question_gen.txt",
        }
        prompts = {}
        for prompt_name, prompt_file in prompt_files.items():
            prompt_path = Path(prompt_file)
            if not prompt_path.is_file():
                raise SystemExit(f"Prompt file not found: {prompt_path}")
            prompts[prompt_name] = prompt_path.read_text()

        pipeline = TemporalPipeline(MODE_CONFIG[mode], **prompts)
        pipeline.run(stage=args.stage)

    elif mode == BenchmarkMode.SPATIAL:
        print("Spatial pipeline not implemented yet.")

    elif mode == BenchmarkMode.COUNTING:
        print("Counting pipeline not implemented yet.")

    else:
        print(f"Unknown mode: {mode}, enter a valid mode 1-4")
        sys.exit(1)


if __name__ == "__main__":
    main()
