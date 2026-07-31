import os
import json
import math
import argparse
from decord import VideoReader, cpu
from PIL import Image


class VideoParser:
    """
    Take a video and segregate it into overlapping sections, then extract
    sampled frames from each section into its own subfolder:

        <frames_root>/<video_name>/<video_name>_section_XXXX/frame_XXXXX.jpg

    No timestamps are burned onto the frames. The folder structure
    (section_id + frame filename) is what lets a downstream VLM report
    "section 3, frame_00042" so we can go back and verify against the file.

    """

    def __init__(self, video_path: str, section_duration: int = 300,
                 overlap: int = 10, target_fps: float = 1.0,
                 frame_width: int = 768, quality: int = 85,
                 frames_root: str = "./frames"):

        # Sanity checks
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if section_duration <= overlap:
            raise ValueError("overlap must be less than section_duration")
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")

        self.video_path = os.path.abspath(video_path)
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.section_duration = section_duration
        self.overlap = overlap
        self.target_fps = target_fps
        self.frame_width = frame_width
        self.quality = quality
        self.frames_root = frames_root

        info = self.get_video_info(self.video_path)
        self.fps = info["fps"]
        self.total_frames = info["total_frames"]
        self.total_duration = info["total_duration"]

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Probe a video with Decord.
        Returns: fps, total_frames, total_duration (seconds).
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        del vr
        return {
            "fps": fps,
            "total_frames": total_frames,
            "total_duration": total_frames / fps if fps else 0.0,
        }

    def video_dir(self) -> str:
        """Top-level folder for this video: <frames_root>/<video_name>."""
        return os.path.join(self.frames_root, self.video_name)

    def section_dir(self, section_id: int) -> str:
        """Per-section subfolder: <video_dir>/<video_name>_section_XXXX."""
        return os.path.join(self.video_dir(),
                            f"{self.video_name}_section_{section_id:04d}")

    def build_sections(self) -> list:
        """
        Compute overlapping frame-index ranges for the whole video.
        No video files are split or re-encoded - only frame ranges are computed.
        """
        step_duration = self.section_duration - self.overlap
        step_frames = int(step_duration * self.fps)
        section_frames = int(self.section_duration * self.fps)

        num_sections = max(
            1, math.ceil((self.total_frames - int(self.overlap * self.fps)) / step_frames)
        )

        sections = []
        for i in range(num_sections):
            start_frame = i * step_frames
            if start_frame >= self.total_frames:
                break
            end_frame = min(start_frame + section_frames, self.total_frames)
            sections.append({
                "section_id": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_global": start_frame / self.fps,
                "end_time_global": end_frame / self.fps,
                "duration": (end_frame - start_frame) / self.fps,
                "frame_dir": self.section_dir(i),
            })
        return sections

    def write_manifest(self, output_path: str, sections: list = None) -> dict:
        """Serialise the section plan to JSON (frame ranges + paths)."""
        if sections is None:
            sections = self.build_sections()
        manifest = {
            "original_video": self.video_path,
            "video_name": self.video_name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "total_duration": self.total_duration,
            "section_duration": self.section_duration,
            "overlap": self.overlap,
            "target_fps": self.target_fps,
            "frame_width": self.frame_width,
            "frames_root": self.frames_root,
            "num_sections": len(sections),
            "sections": sections,
        }
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def extract_section_frames(self, section: dict) -> list:
        """
        Sample frames for one section at target_fps, resize to frame_width,
        and save them into the section's subfolder as frame_XXXXX.jpg.

        Returns the list of saved frame file paths (chronological order).
        """
        section_id = section["section_id"]
        start_frame = section["start_frame"]
        end_frame = section["end_frame"]

        frames_dir = self.section_dir(section_id)
        os.makedirs(frames_dir, exist_ok=True)

        vr = VideoReader(self.video_path, ctx=cpu(0))
        step = max(1, int(self.fps / self.target_fps))
        frame_indices = list(range(start_frame, end_frame, step))

        frame_paths = []
        for i, frame_idx in enumerate(frame_indices):
            frame = vr[frame_idx].asnumpy()
            img = Image.fromarray(frame)
            img = img.resize(
                (self.frame_width, int(img.height * self.frame_width / img.width)),
                Image.LANCZOS,
            )
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
            img.save(frame_path, quality=self.quality)
            frame_paths.append(frame_path)

        if not frame_paths:
            raise RuntimeError(
                f"No frames extracted from {self.video_path} "
                f"(range {start_frame}-{end_frame}, fps={self.fps})"
            )
        return frame_paths

    def prepare_sections(self, manifest_path: str = None) -> list:
        """
        End-to-end: build sections, optionally write a manifest, and extract
        every section's frames into its own subfolder.
        Returns a list of {section, frames_dir, frame_paths} dicts.
        """
        sections = self.build_sections()
        if manifest_path:
            self.write_manifest(manifest_path, sections)

        results = []
        for section in sections:
            frame_paths = self.extract_section_frames(section)
            results.append({
                "section": section,
                "frames_dir": self.section_dir(section["section_id"]),
                "frame_paths": frame_paths,
            })
            print(f"{self.video_name}_section_{section['section_id']:04d}: "
                  f"{len(frame_paths)} frames -> {self.section_dir(section['section_id'])}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segregate a video into per-section frame subfolders (no AI calls)."
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--manifest", default=None,
                        help="Optional path to write the section manifest JSON")
    parser.add_argument("--frames-root", default="./frames",
                        help="Root dir for extracted frame folders (default: ./frames)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Duration of each section in seconds (default: 300)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Overlap between sections in seconds (default: 10)")
    parser.add_argument("--target-fps", type=float, default=1.0,
                        help="Frames sampled per second of video (default: 1.0)")
    parser.add_argument("--frame-width", type=int, default=768,
                        help="Width frames are resized to before saving (px)")

    args = parser.parse_args()

    vp = VideoParser(
        video_path=args.video_path,
        section_duration=args.duration,
        overlap=args.overlap,
        target_fps=args.target_fps,
        frame_width=args.frame_width,
        frames_root=args.frames_root,
    )
    vp.prepare_sections(manifest_path=args.manifest)