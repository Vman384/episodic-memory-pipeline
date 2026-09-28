"""Convert timestamped frame timeframes into elapsed video seconds."""

import re
from pathlib import Path

MICROSECONDS_PER_SECOND = 1_000_000

# Matches plain epoch-microsecond names (Boreas) and WildScenes-style
# "<epoch-seconds>-<nanoseconds>" names.
_FRAME_TIMESTAMP_PATTERN = re.compile(r"^(\d+)(?:-(\d{1,9}))?$")


def parse_frame_timestamp(frame):
    """Return a frame filename's epoch timestamp in microseconds.

    Supports plain epoch-microsecond filenames (Boreas) and
    ``<epoch-seconds>-<nanoseconds>`` filenames (WildScenes).
    """
    match = _FRAME_TIMESTAMP_PATTERN.match(Path(str(frame)).stem)
    if not match:
        raise ValueError(
            "Frame must have a timestamp filename (e.g. 1733343593917869.png "
            f"or 1624328055-542742850.png): {frame}"
        )
    seconds, nanoseconds = match.groups()
    if nanoseconds is None:
        return int(seconds)
    return int(seconds) * MICROSECONDS_PER_SECOND + int(nanoseconds.ljust(9, "0")) // 1000


class TimeframeConverter:
    """Convert frame timestamps to seconds relative to the first video frame."""

    def __init__(self, frames_dir):
        self.frames_dir = Path(frames_dir).expanduser()
        frame_timestamps = self._frame_timestamps()
        if not frame_timestamps:
            raise ValueError(f"No timestamped frames found in {self.frames_dir}")
        self.video_start_timestamp = frame_timestamps[0]

    def _frame_timestamps(self):
        """
        Return frame timestamps in chronological order.
        """
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {self.frames_dir}")

        timestamps = []
        for frame_path in self.frames_dir.iterdir():
            if not frame_path.is_file():
                continue
            try:
                timestamps.append(parse_frame_timestamp(frame_path))
            except ValueError:
                continue
        return sorted(timestamps)

    def frame_to_seconds(self, frame):
        """Return a frame's elapsed time in seconds from the video start."""
        timestamp = parse_frame_timestamp(frame)
        return (timestamp - self.video_start_timestamp) / MICROSECONDS_PER_SECOND

    def timeframe_to_seconds(self, start_frame, end_frame):
        """Return the elapsed start and end seconds for an event timeframe."""
        start_seconds = self.frame_to_seconds(start_frame)
        end_seconds = self.frame_to_seconds(end_frame)
        if end_seconds < start_seconds:
            raise ValueError("end_frame must not occur before start_frame")
        return {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        }
