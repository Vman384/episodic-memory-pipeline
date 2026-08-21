"""Convert timestamped frame timeframes into elapsed video seconds."""

from pathlib import Path


class TimeframeConverter:
    """Convert frame timestamps to seconds relative to the first video frame."""

    MICROSECONDS_PER_SECOND = 1_000_000

    def __init__(self, frames_dir):
        self.frames_dir = Path(frames_dir).expanduser()
        frame_timestamps = self._frame_timestamps()
        if not frame_timestamps:
            raise ValueError(f"No timestamped frames found in {self.frames_dir}")
        self.video_start_timestamp = frame_timestamps[0]

    def _frame_timestamps(self):
        """
        Return numeric frame timestamps in chronological order.
        """
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {self.frames_dir}")

        timestamps = []
        for frame_path in self.frames_dir.iterdir():
            if not frame_path.is_file():
                continue
            try:
                timestamps.append(int(frame_path.stem))
            except ValueError:
                continue
        return sorted(timestamps)

    @staticmethod
    def _timestamp(frame):
        """Extract the numeric timestamp from a frame filename."""
        try:
            return int(Path(str(frame)).stem)
        except ValueError as error:
            raise ValueError(f"Frame must have a numeric filename: {frame}") from error

    def frame_to_seconds(self, frame):
        """Return a frame's elapsed time in seconds from the video start."""
        timestamp = self._timestamp(frame)
        return (timestamp - self.video_start_timestamp) / self.MICROSECONDS_PER_SECOND

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
