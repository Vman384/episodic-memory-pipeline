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

    args = parser.parse_args()

    mode = BenchmarkMode(args.mode)

    if mode == BenchmarkMode.SPARSE_EVENT:
        from pipeline.SparseEventPipeline import SparseEventPipeline
        config_path = MODE_CONFIG[mode]
        pipeline = SparseEventPipeline(config_path)
        pipeline.run()

    elif mode == BenchmarkMode.TEMPORAL:
        print("Temporal pipeline not implemented yet.")

    elif mode == BenchmarkMode.SPATIAL:
        print("Spatial pipeline not implemented yet.")

    elif mode == BenchmarkMode.COUNTING:
        print("Counting pipeline not implemented yet.")

    else:
        print(f"Unknown mode: {mode}, enter a valid mode 1-4")
        sys.exit(1)


if __name__ == "__main__":
    main()
