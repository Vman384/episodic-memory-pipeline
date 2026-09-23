"""Split a directory of video frames into smaller VLM-sized batches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from pipeline.timeframe_converter import TimeframeConverter
except ImportError:
    from timeframe_converter import TimeframeConverter

try:
    from pipeline.image_preprocessor import ImagePreprocessor
except ImportError:
    from image_preprocessor import ImagePreprocessor


class FrameParser:
    """Partition video frames into numbered section directories.

    The parser expects one directory containing frame image files.  It copies
    (or optionally moves) sampled frames into groups of at most
    ``frames_per_section``::

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
        frames_per_section: Number of frames sampled into each section.  The
            final section may contain fewer frames.
        step_size: Stride between sampled frames inside one section.  A value
            of 1 samples consecutive frames, while 3 samples the 1st, 4th,
            7th, and so on, until the section is full.
        skip: Number of frames skipped after a section's sampling stride
            before the next section starts.  The next section begins at
            ``start + frames_per_section * step_size + skip``, so 0 continues
            sampling as one uninterrupted stride.
        move: Move frames instead of copying them.
        preprocessor: Optional ``ImagePreprocessor`` re-encodes each sampled
            frame as a resized JPEG while writing sections.  When omitted,
            frames are copied (or moved) untouched.
        extensions: Accepted image extensions.  Matching is case-insensitive.
    """

    DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(
        self,
        frames_dir: str | Path,
        output_dir: str | Path | None = None,
        frames_per_section: int = 100,
        step_size: int = 1,
        skip: int = 0,
        move: bool = False,
        preprocessor=None,
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
        if skip < 0:
            raise ValueError("skip must be zero or greater")
        if not extensions:
            raise ValueError("extensions must contain at least one image extension")

        self.frames_per_section = frames_per_section
        self.step_size = step_size
        self.skip = skip
        self.move = move
        self.preprocessor = preprocessor
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
        frame_paths = self._get_frame_paths()
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
        stride_window = self.frames_per_section * self.step_size
        section_index = 0
        section_start = 0
        while section_start < len(frame_paths):

            section_frames = frame_paths[section_start : section_start + stride_window : self.step_size]
            section_dir = self.output_dir / f"section_{section_index:04d}"
            section_dir.mkdir(parents=True, exist_ok=True)
            section_dirs.append(section_dir)

            for frame_path in section_frames:
                destination = section_dir / frame_path.name
                if self.preprocessor is not None:
                    destination = section_dir / f"{frame_path.stem}{self.preprocessor.SUFFIX}"
                    self.preprocessor.process(frame_path, destination)
                    if self.move:
                        frame_path.unlink()
                elif self.move:
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

            section_start += stride_window + self.skip
            section_index += 1

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
        skip=config.get("skip", 0),
        move=config.get("move", False),
        preprocessor=ImagePreprocessor.from_config(config),
    ).create_section_dir()

    for section_dir in section_dirs:
        print(section_dir)

if __name__ == "__main__":
    main()
