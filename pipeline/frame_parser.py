"""Split a directory of video frames into smaller VLM-sized batches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image


try:
    from pipeline.timeframe_converter import TimeframeConverter
except ImportError:
    from timeframe_converter import TimeframeConverter


class FrameParser:
    """Partition video frames into numbered section directories.

    The parser expects one directory containing frame image files.  It copies
    (or optionally moves) those files into groups of ``frames_per_section``::

        input_frames/
            1733343593917869.png
            1733343596067826.png
            ...

        output_frames/
            section_0000/1733343593917869.png
            section_0000/1733343596067826.png
            section_0001/1733343656170615.png
            ...

    Args:
        frames_dir: Directory containing the input frame files.
        output_dir: Directory in which section directories are created.  When
            omitted, a sibling directory named ``<frames_dir>_sections`` is
            used.
        frames_per_section: Maximum number of frames in each section.
        step_size: Keep every ``step_size``-th frame after sorting.  A value
            of 1 keeps every frame, while 2 keeps frames 0, 2, 4, and so on.
        move: Move frames instead of copying them.
        extensions: Accepted image extensions.  Matching is case-insensitive.
    """

    DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(
        self,
        frames_dir: str | Path,
        output_dir: str | Path | None = None,
        frames_per_section: int = 100,
        step_size: int = 1,
        move: bool = False,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS):

        # Get the path of all the frames
        self.frames_dir = Path(frames_dir).expanduser()

        self.output_dir = (
            Path(output_dir).expanduser()
            if output_dir is not None
            else self.frames_dir.parent / f"{self.frames_dir.name}_sections"
        )

        if frames_per_section <= 0:
            raise ValueError("frames_per_section must be greater than zero")
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero")
        if not extensions:
            raise ValueError("extensions must contain at least one image extension")

        self.frames_per_section = frames_per_section
        self.step_size = step_size
        self.move = move
        normalized_extensions = set()
        for extension in extensions:
            if extension.startswith("."):
                normalized_extensions.add(extension.lower())
            else:
                normalized_extensions.add(f".{extension.lower()}")
        self.extensions = frozenset(normalized_extensions)

    def _get_frame_paths(self) -> list[Path]:
        """Return supported frame files in numeric filename order."""
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {self.frames_dir}")

        frame_paths = []
        for path in self.frames_dir.iterdir():
            if path.is_file() and path.suffix.lower() in self.extensions:
                frame_paths.append(path)
        return sorted(frame_paths, key=lambda path: int(path.stem))

    def create_section_dir(self) -> list[Path]:
        """Create section directories and return their paths.

        Existing section directories are reused.  Existing files with the
        same names are replaced, making repeated runs deterministic.  An
        empty input directory returns an empty list without creating output.

        A ``sections.json`` manifest is written next to the section
        directories, mapping each section to its first/last frame and elapsed
        start/end seconds relative to the first sampled frame of the video.
        """
        frame_paths = self._get_frame_paths()[:: self.step_size]
        if not frame_paths:
            return []

        # sanity check to ensure output directory is different from frame directory
        if self.output_dir.resolve() == self.frames_dir.resolve():
            raise ValueError("output_dir must be different from frames_dir")

        # make a new output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        section_dirs: list[Path] = []
        section_manifest = {}

        # Reuse the shared converter for all per-section second math.
        converter = TimeframeConverter(self.frames_dir)
        video_start_timestamp = converter.video_start_timestamp
        for section_index, start in enumerate(range(0, len(frame_paths), self.frames_per_section)):

            section_frames = frame_paths[start : start + self.frames_per_section]
            section_dir = self.output_dir / f"section_{section_index:04d}"
            section_dir.mkdir(parents=True, exist_ok=True)
            section_dirs.append(section_dir)

            for frame_path in section_frames:
                destination = section_dir / frame_path.name
                # Depending on params, either we move the actual frames themselves or we make a copy
                if self.move:
                    shutil.move(str(frame_path), str(destination))
                else:
                    shutil.copy2(frame_path, destination)

            # Per-section elapsed seconds relative to the first sampled frame.
            seconds = converter.timeframe_to_seconds(
                section_frames[0].name,
                section_frames[-1].name,
            )
            section_manifest[f"section_{section_index:04d}"] = {
                "start_frame": section_frames[0].name,
                "end_frame": section_frames[-1].name,
                "start_seconds": seconds["start_seconds"],
                "end_seconds": seconds["end_seconds"],
            }

        manifest_path = self.output_dir / "sections.json"
        with open(manifest_path, "w") as manifest_file:
            json.dump(
                {
                    "frames_dir": str(self.frames_dir),
                    "video_start_timestamp": video_start_timestamp,
                    "sections": section_manifest,
                },
                manifest_file,
                indent=2,
            )

        return section_dirs


def main() -> None:
    """Run the frame partitioner from a JSON configuration file."""
    parser = argparse.ArgumentParser(
        description="Split a directory of video frames into VLM-sized sections."
    )
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    section_dirs = FrameParser(
        frames_dir=config["frames_dir"],
        output_dir=config["sections_dir"],
        frames_per_section=config["frames_per_section"],
        step_size=config["step"],
        move=config.get("move", False),
    ).create_section_dir()

    for section_dir in section_dirs:
        print(section_dir)

if __name__ == "__main__":
    main()


def format_timestamp(seconds):
    """Converts seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def sample_frames_from_indices(vr, start_idx, end_idx, num_samples=8, max_size=(448, 448)):
    """Uniformly samples frames and resizes them to drastically cut visual token usage."""
    total_chunk_frames = end_idx - start_idx
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    if total_chunk_frames <= 0:
        return []

    sample_count = min(num_samples, total_chunk_frames)
    if sample_count == 1:
        selected_indices = [start_idx]
    else:
        selected_indices = [
            start_idx + round(index * (total_chunk_frames - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    
    batch = vr.get_batch(selected_indices).asnumpy()
    frames = []
    for frame in batch:
        img = Image.fromarray(frame)
        # Downscale image preserving aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        frames.append(img)
    return frames 