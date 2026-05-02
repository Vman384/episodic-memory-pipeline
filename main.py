#!/usr/bin/env python3
"""
Sparse Event Localisation Annotation Pipeline
==============================================
Splits a video into 30-second clips, runs SAM2 automatic segmentation on
keyframes, uses Claude Vision to identify detected objects, then generates
Sparse Event Localisation Q&A pairs (verifying existence of rare/outlier events).

Output schema per video:
{
  "video": str,
  "total_clips": int,
  "clip_duration_seconds": int,
  "annotation_category": "sparse_event_localisation",
  "clip_analysis": [ { clip_name, unique_objects, frame_details } ],
  "qa_pairs": [
    {
      "question": str,
      "answer": str,
      "event_present": bool,
      "relevant_clips": [str]
    }
  ]
}

Requirements:
    pip install anthropic opencv-python pillow torch numpy
    pip install git+https://github.com/facebookresearch/segment-anything-2.git

SAM2 checkpoints:  https://github.com/facebookresearch/segment-anything-2#model-checkpoints
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
from pathlib import Path
from typing import Optional

import anthropic
import cv2
import numpy as np
import torch
from PIL import Image

# SAM2 imports (install from: https://github.com/facebookresearch/segment-anything-2)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_DURATION_SEC = 30
DEFAULT_FRAMES_PER_CLIP = 3          # keyframes sampled per clip
DEFAULT_SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
RARE_OBJECT_THRESHOLD = 0.20         # object present in <20 % of clips → rare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def frame_to_base64_png(frame: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64-encoded PNG string."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def colorise_masks(frame: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Overlay semi-transparent coloured masks on a frame copy."""
    overlay = frame.copy().astype(np.float32)
    rng = np.random.default_rng(seed=42)
    for mask in masks:
        colour = rng.integers(60, 230, size=3).astype(np.float32)
        seg: np.ndarray = mask["segmentation"]
        overlay[seg] = overlay[seg] * 0.45 + colour * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class SparseEventAnnotationPipeline:
    """
    End-to-end pipeline:
      1. Split video → 30-second clips (ffmpeg).
      2. Extract keyframes → run SAM2 automatic mask generation.
      3. Send (original + masked) frames to Claude Vision → object list.
      4. Aggregate objects per clip; flag rare ones.
      5. Call Claude to generate Sparse Event Localisation Q&A.
      6. Write results to JSON.
    """

    def __init__(
        self,
        sam2_checkpoint: str = DEFAULT_SAM2_CHECKPOINT,
        sam2_model_cfg: str = DEFAULT_SAM2_CFG,
        anthropic_api_key: Optional[str] = None,
        frames_per_clip: int = DEFAULT_FRAMES_PER_CLIP,
        output_dir: str = "output",
        clips_dir: str = "clips",
    ) -> None:
        self.frames_per_clip = frames_per_clip
        self.output_dir = Path(output_dir)
        self.clips_dir = Path(clips_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        # Anthropic client
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

        # SAM2 setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[init] Using device: {self.device}")
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=16,          # lower = faster, coarser masks
            pred_iou_thresh=0.80,
            stability_score_thresh=0.92,
            min_mask_region_area=500,    # ignore tiny noise regions
        )
        print("[init] SAM2 mask generator ready.\n")

    # ------------------------------------------------------------------
    # Step 1 – Video splitting
    # ------------------------------------------------------------------

    def split_video(
        self, video_path: str | Path, clip_duration: int = CLIP_DURATION_SEC
    ) -> list[Path]:
        """
        Split *video_path* into consecutive clips of *clip_duration* seconds.
        Returns a list of clip Paths sorted by start time.
        """
        video_path = Path(video_path)
        duration = get_video_duration(video_path)
        print(
            f"[split] Duration: {duration:.1f}s  →  "
            f"{int(duration / clip_duration) + 1} clips of {clip_duration}s"
        )

        clip_paths: list[Path] = []
        start = 0.0
        idx = 0

        while start < duration:
            clip_path = self.clips_dir / f"clip_{idx:04d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-an",              # drop audio – not needed
                    str(clip_path),
                ],
                capture_output=True,
                check=True,
            )
            clip_paths.append(clip_path)
            start += clip_duration
            idx += 1

        print(f"[split] Created {len(clip_paths)} clips in '{self.clips_dir}'\n")
        return clip_paths

    # ------------------------------------------------------------------
    # Step 2 – Keyframe extraction
    # ------------------------------------------------------------------

    def extract_keyframes(self, clip_path: Path) -> list[np.ndarray]:
        """
        Return *self.frames_per_clip* evenly-spaced RGB frames from the clip.
        """
        cap = cv2.VideoCapture(str(clip_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []

        indices = np.linspace(0, total - 1, self.frames_per_clip, dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    # ------------------------------------------------------------------
    # Step 3 – SAM2 segmentation
    # ------------------------------------------------------------------

    def run_sam2(self, frame: np.ndarray) -> list[dict]:
        """Run SAM2 automatic mask generation on a single RGB frame."""
        return self.mask_generator.generate(frame)

    # ------------------------------------------------------------------
    # Step 4 – Object identification via Claude Vision
    # ------------------------------------------------------------------

    def identify_objects(
        self, frame: np.ndarray, masks: list[dict]
    ) -> list[str]:
        """
        Send original frame + mask overlay to Claude Vision.
        Returns a deduplicated list of lowercase object names.
        """
        original_b64 = frame_to_base64_png(frame)
        overlay_b64 = frame_to_base64_png(colorise_masks(frame, masks))

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an object-detection assistant. "
                                "Image 1 is an original video frame. "
                                "Image 2 shows coloured SAM2 segmentation masks "
                                "overlaid on the same frame — each colour highlights "
                                "a distinct object or region.\n\n"
                                "List EVERY unique object / entity visible in this "
                                "scene, paying special attention to anything unusual "
                                "or unexpected.\n\n"
                                "Reply with ONLY a JSON array of lowercase strings. "
                                "No markdown fences, no explanations.\n"
                                'Example: ["road", "car", "cyclist", "deer"]'
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": original_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": overlay_b64,
                            },
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip().replace("```json", "").replace("```", "")
        try:
            return [str(o).lower().strip() for o in json.loads(raw)]
        except json.JSONDecodeError:
            return []

    # ------------------------------------------------------------------
    # Step 5 – Per-clip aggregation
    # ------------------------------------------------------------------

    def process_clip(self, clip_path: Path) -> dict:
        """
        Extract keyframes → SAM2 → object identification.
        Returns aggregated clip-level data.
        """
        print(f"  [clip] {clip_path.name} ...", end=" ", flush=True)
        frames = self.extract_keyframes(clip_path)
        unique_objects: set[str] = set()
        frame_details: list[dict] = []

        for i, frame in enumerate(frames):
            masks = self.run_sam2(frame)
            objects = self.identify_objects(frame, masks)
            unique_objects.update(objects)
            frame_details.append(
                {
                    "frame_index": i,
                    "num_masks": len(masks),
                    "detected_objects": objects,
                }
            )

        print(f"objects: {sorted(unique_objects)}")
        return {
            "clip_name": clip_path.name,
            "unique_objects": sorted(unique_objects),
            "frame_details": frame_details,
        }

    # ------------------------------------------------------------------
    # Step 6 – Q&A generation
    # ------------------------------------------------------------------

    def generate_qa(
        self, video_path: str | Path, clip_data: list[dict]
    ) -> list[dict]:
        """
        Given per-clip object data, ask Claude to generate Sparse Event
        Localisation Q&A pairs.
        """
        # Build object → clips mapping
        obj_clip_map: dict[str, list[str]] = {}
        for clip in clip_data:
            for obj in clip["unique_objects"]:
                obj_clip_map.setdefault(obj, []).append(clip["clip_name"])

        total_clips = len(clip_data)
        rare_objects = [
            obj
            for obj, clips in obj_clip_map.items()
            if len(clips) / total_clips < RARE_OBJECT_THRESHOLD
        ]

        context = json.dumps(
            {
                "video_file": Path(video_path).name,
                "total_clips": total_clips,
                "clip_duration_seconds": CLIP_DURATION_SEC,
                "all_detected_objects": sorted(obj_clip_map.keys()),
                "rare_objects_lt_20pct_clips": sorted(rare_objects),
                "object_to_clips_mapping": obj_clip_map,
            },
            indent=2,
        )

        prompt = f"""You are building a video-understanding annotation dataset.
Category: **Sparse Event Localisation** — questions that verify whether a rare,
outlier, or unexpected event occurred at ANY point in a video.

Rules:
- Each question must ask whether a specific rare/unusual entity or event was
  present *at any point* during the full video.
- Include a mix of POSITIVE questions (event DID occur) and NEGATIVE questions
  (plausible event that did NOT occur — pick something believable but absent).
- Each answer must state whether the event was observed and name the specific
  clip(s) where it was/wasn't found.
- Generate between 5 and 8 Q&A pairs.

Detection data:
{context}

Respond with ONLY a JSON array. Each element must have these keys:
  "question"        – string
  "answer"          – string
  "event_present"   – boolean
  "relevant_clips"  – array of clip filename strings

No markdown, no preamble, no explanation outside the JSON array.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = (
            response.content[0].text.strip()
            .replace("```json", "")
            .replace("```", "")
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("[warn] Could not parse Q&A JSON from model response.")
            return []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self, video_path: str | Path, clip_duration: int = CLIP_DURATION_SEC
    ) -> dict:
        """
        Execute the full pipeline and write results to
        <output_dir>/<video_stem>_annotations.json.
        Returns the output dict.
        """
        video_path = Path(video_path)
        print("=" * 60)
        print("  Sparse Event Localisation Annotation Pipeline")
        print(f"  Video : {video_path.name}")
        print("=" * 60)

        # ── 1. Split ──────────────────────────────────────────────────
        print("\n[1/3] Splitting video into clips …")
        clip_paths = self.split_video(video_path, clip_duration)

        # ── 2. Segment & identify objects ─────────────────────────────
        print("[2/3] Running SAM2 + object identification per clip …")
        all_clip_data: list[dict] = [
            self.process_clip(cp) for cp in clip_paths
        ]

        # ── 3. Generate Q&A ───────────────────────────────────────────
        print("\n[3/3] Generating Sparse Event Localisation Q&A …")
        qa_pairs = self.generate_qa(video_path, all_clip_data)
        print(f"  Generated {len(qa_pairs)} Q&A pairs.")

        # ── Compile & save ────────────────────────────────────────────
        output = {
            "video": video_path.name,
            "total_clips": len(clip_paths),
            "clip_duration_seconds": clip_duration,
            "annotation_category": "sparse_event_localisation",
            "clip_analysis": all_clip_data,
            "qa_pairs": qa_pairs,
        }

        out_path = self.output_dir / f"{video_path.stem}_annotations.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n✓ Annotations saved → {out_path}\n")
        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse Event Localisation annotation pipeline."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "--sam2-checkpoint",
        default=DEFAULT_SAM2_CHECKPOINT,
        help=f"SAM2 .pt checkpoint path (default: {DEFAULT_SAM2_CHECKPOINT})",
    )
    parser.add_argument(
        "--sam2-cfg",
        default=DEFAULT_SAM2_CFG,
        help=f"SAM2 YAML config path (default: {DEFAULT_SAM2_CFG})",
    )
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=CLIP_DURATION_SEC,
        help=f"Clip length in seconds (default: {CLIP_DURATION_SEC})",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=DEFAULT_FRAMES_PER_CLIP,
        help=f"Keyframes sampled per clip (default: {DEFAULT_FRAMES_PER_CLIP})",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for the output JSON (default: output/)",
    )
    parser.add_argument(
        "--clips-dir",
        default="clips",
        help="Directory for intermediate clip files (default: clips/)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var).",
    )
    args = parser.parse_args()

    pipeline = SparseEventAnnotationPipeline(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_cfg,
        anthropic_api_key=args.api_key,
        frames_per_clip=args.frames_per_clip,
        output_dir=args.output_dir,
        clips_dir=args.clips_dir,
    )
    pipeline.run(args.video_path, clip_duration=args.clip_duration)


if __name__ == "__main__":
    main()