"""Split a directory of video frames into smaller VLM-sized batches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


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

    Original filenames are retained so downstream code can refer to the
    source frame unambiguously.  Frame filenames must have numeric stems,
    which are used for chronological ordering.

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
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ):

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
        self.extensions = frozenset(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in extensions
        )

    def _get_frame_paths(self) -> list[Path]:
        """Return supported frame files in numeric filename order."""
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {self.frames_dir}")

        frame_paths = [
            path
            for path in self.frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        return sorted(frame_paths, key=lambda path: int(path.stem))

    def create_section_dir(self) -> list[Path]:
        """Create section directories and return their paths.

        Existing section directories are reused.  Existing files with the
        same names are replaced, making repeated runs deterministic.  An
        empty input directory returns an empty list without creating output.
        """
        frame_paths = self._get_frame_paths()[:: self.step_size]
        if not frame_paths:
            return []

        if self.output_dir.resolve() == self.frames_dir.resolve():
            raise ValueError("output_dir must be different from frames_dir")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        section_dirs: list[Path] = []

        for section_index, start in enumerate(
            range(0, len(frame_paths), self.frames_per_section)
        ):
            section_dir = self.output_dir / f"section_{section_index:04d}"
            section_dir.mkdir(parents=True, exist_ok=True)
            section_dirs.append(section_dir)

            for frame_path in frame_paths[start : start + self.frames_per_section]:
                destination = section_dir / frame_path.name
                if self.move:
                    shutil.move(str(frame_path), str(destination))
                else:
                    shutil.copy2(frame_path, destination)

        return section_dirs


def main() -> None:
    
    """Run the frame partitioner from the command line."""
    parser = argparse.ArgumentParser(
        description="Split a directory of video frames into VLM-sized sections."
    )
    parser.add_argument("frames_dir", help="Directory containing frame images")

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <frames_dir>_sections)",
    )
    parser.add_argument(
        "--frames-per-section",
        type=int,
        default=100,
        help="Maximum frames per section (default: 100)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Keep every Nth frame (default: 1, keep every frame)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move frames instead of copying them",
    )
    args = parser.parse_args()

    section_dirs = FrameParser(
        frames_dir=args.frames_dir,
        output_dir=args.output,
        frames_per_section=args.frames_per_section,
        step_size=args.step,
        move=args.move,
    ).create_section_dir()

    for section_dir in section_dirs:
        print(section_dir)

if __name__ == "__main__":
    main()
