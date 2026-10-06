import os

# Must be set BEFORE importing transformers, otherwise they have no effect.
os.environ.setdefault("HF_HOME", "/scratch/pg06/vm4618/huggingface_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import re
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Sam3Model,
    Sam3Processor,
)

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import CountingPipelineConfig
from pipeline.frame_parser import FrameParser, format_timestamp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_frame_paths(folder: Path) -> list[Path]:
    """Frame files in a section folder, in numeric (timestamp) filename order."""
    paths = [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def sample_frames_from_paths(
    paths: list[Path], num_samples: int = 16, max_size: tuple[int, int] = (640, 640)
) -> list[Image.Image]:
    """Uniformly pick `num_samples` frame files, load them and downscale (aspect preserved)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    if not paths:
        return []

    count = min(num_samples, len(paths))
    if count == 1:
        indices = [0]
    else:
        indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]

    frames = []
    for idx in indices:
        with Image.open(paths[idx]) as img:
            frame = img.convert("RGB")
        frame.thumbnail(max_size, Image.Resampling.LANCZOS)
        frames.append(frame)
    return frames


class CountingPipeline:
    """VLM concept discovery -> SAM 3 text-prompted segmentation -> CLIP embedding dedup.

    "Count" here means: number of distinct object instances seen over the whole video.

    Config keys used (new ones are optional and have defaults):
        frames_dir, output, frames_per_section, num_samples_per_section, max_frame_size,
        section_overlap (0.1)       fraction of a section shared with the next one
        sam3_threshold (0.5)        SAM 3 instance score threshold
        min_mask_area_frac (0.001)  drop masks smaller than this fraction of the image
        drop_edge_touching (False)  drop masks that touch the image border
        crop_padding (0.1)          context added around each box before embedding
        label_synonyms ({})         e.g. {"sedan": "car", "pedestrian": "person"}
        save_crops (True)           save a crop of every counted instance for debugging
    """

    def __init__(
        self,
        config: CountingPipelineConfig,
        prompt: str,
        frame_parser: FrameParser,
        ai_parser: AIParser,
        similarity_threshold: float = 0.85,
    ):
        self.prompt = prompt
        self.config = config
        self.frame_parser = frame_parser
        self.ai_parser = ai_parser
        self.similarity_threshold = similarity_threshold

        self.sam3_threshold = config.get("sam3_threshold", 0.5)
        self.min_mask_area_frac = config.get("min_mask_area_frac", 0.001)
        self.drop_edge_touching = config.get("drop_edge_touching", False)
        self.crop_padding = config.get("crop_padding", 0.1)
        self.label_synonyms = {
            k.lower(): v.lower() for k, v in (config.get("label_synonyms") or {}).items()
        }
        self.save_crops = config.get("save_crops", True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # State
        self.embedding_memory: dict[str, torch.Tensor] = {}  # label -> (N, D) normalized
        self.object_counts: dict[str, int] = {}
        self.all_results: list[dict] = []

        # CLIP
        clip_model_name = config.get("CLIPModelmodel", "openai/clip-vit-base-patch32")
        clip_proc_name = config.get("CLIPProcessormodel", "openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_proc_name)
        self.clip_model.eval()

        # Detector: "sam3" (default) or "grounding_dino"
        self.detector = config.get("detector", "sam3")
        if self.detector == "sam3":
            sam3_model_name = config.get("sam3_model", "facebook/sam3")
            self.sam3_model = Sam3Model.from_pretrained(sam3_model_name).to(self.device)
            self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_name)
            self.sam3_model.eval()
        elif self.detector == "grounding_dino":
            gd_name = config.get("grounding_dino_model", "IDEA-Research/grounding-dino-base")
            self.gd_processor = AutoProcessor.from_pretrained(gd_name)
            self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_name).to(self.device)
            self.gd_model.eval()
            self.gd_box_threshold = config.get("gd_box_threshold", 0.35)
            self.gd_text_threshold = config.get("gd_text_threshold", 0.25)
        else:
            raise ValueError("detector must be 'sam3' or 'grounding_dino'")

    # ------------------------------------------------------------------ utils
    def write_json_output(self, output_dir: Path, file_name: str, output: dict | list) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / file_name, "w") as f:
            json.dump(output, f, indent=2)

    def normalize_label(self, label: str) -> str:
        label = re.sub(r"\s+", " ", str(label).strip().lower())
        return self.label_synonyms.get(label, label)

    def parse_concept_list(self, vlm_response: str | list) -> list[str]:
        """Parse VLM output into a de-duplicated, order-preserving list of normalized labels."""
        if isinstance(vlm_response, list):
            items = vlm_response
        else:
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", vlm_response).strip()
            items = None
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                match = re.search(r"\[[\s\S]*?\]", cleaned)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, list):
                            items = parsed
                    except json.JSONDecodeError:
                        pass
            if items is None:
                items = []
                for line in cleaned.split("\n"):
                    line = re.sub(r"^[\s\*\-\d\.]+", "", line).strip()
                    items.extend(c.strip() for c in line.split(",") if c.strip())

        return list(dict.fromkeys(self.normalize_label(i) for i in items if i))

    # -------------------------------------------------------------- embeddings
    @torch.no_grad()
    def get_clip_embeddings(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Batched, L2-normalized CLIP embeddings. Returns (N, D) on CPU."""
        chunks = []
        for i in range(0, len(images), batch_size):
            inputs = self.clip_processor(images=images[i : i + batch_size], return_tensors="pt")
            inputs = inputs.to(self.device)  # type: ignore
            feats = self.clip_model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            elif isinstance(feats, tuple):
                feats = feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.float().cpu())
        return torch.cat(chunks, dim=0)

    def novelty_flags(self, embeddings: torch.Tensor, label: str) -> list[bool]:
        """True for each embedding that is NOT similar to anything already in memory.

        Compared only against memory from earlier frames/sections, never against
        other instances in the same frame (those are distinct by definition).
        """
        memory = self.embedding_memory.get(label)
        if memory is None or len(memory) == 0:
            return [True] * len(embeddings)
        max_sim = (embeddings @ memory.T).max(dim=1).values
        return (max_sim <= self.similarity_threshold).tolist()

    # -------------------------------------------------------------------- SAM 3
    def isolate_objects_with_sam3(
        self, images: list[Image.Image], text_prompt: str
    ) -> list[list[dict]]:
        """Run SAM 3 on all frames for one concept.

        Returns one list per input frame. Each instance is a dict with
        'crop' (bbox crop with padding), 'bbox', 'score'.
        """
        inputs = self.sam3_processor(
            images=images, text=[text_prompt] * len(images), return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3_model(**inputs)

        target_sizes = [img.size[::-1] for img in images]
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.sam3_threshold,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )

        per_frame: list[list[dict]] = []
        for img, res in zip(images, results):
            instances: list[dict] = []
            if "masks" not in res:
                per_frame.append(instances)
                continue

            rgb = img.convert("RGB")
            w, h = rgb.size
            masks = res["masks"].cpu().numpy().astype(bool)
            scores = res["scores"].cpu().numpy() if "scores" in res else [None] * len(masks)

            for mask, score in zip(masks, scores):
                rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
                if not rows.any() or not cols.any():
                    continue
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                # Quality filters: tiny masks and (optionally) truncated objects embed poorly
                if mask.sum() / float(w * h) < self.min_mask_area_frac:
                    continue
                if self.drop_edge_touching and (xmin == 0 or ymin == 0 or xmax >= w - 1 or ymax >= h - 1):
                    continue

                # Crop the box with some real context instead of blacking out the background
                pad_x = int((xmax - xmin) * self.crop_padding)
                pad_y = int((ymax - ymin) * self.crop_padding)
                box = (
                    max(0, int(xmin) - pad_x),
                    max(0, int(ymin) - pad_y),
                    min(w, int(xmax) + pad_x + 1),
                    min(h, int(ymax) + pad_y + 1),
                )
                instances.append(
                    {
                        "crop": rgb.crop(box),
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                        "score": None if score is None else float(score),
                    }
                )
            per_frame.append(instances)
        return per_frame

    # ----------------------------------------------------------- Grounding DINO
    def isolate_objects_with_grounding_dino(
        self, images: list[Image.Image], text_prompt: str
    ) -> list[list[dict]]:
        """Same return format as isolate_objects_with_sam3, but boxes only (no masks)."""
        # Grounding DINO expects lowercase text terminated by a period.
        text = text_prompt.lower().strip()
        if not text.endswith("."):
            text += "."

        inputs = self.gd_processor(
            images=images, text=[text] * len(images), return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.gd_model(**inputs)

        target_sizes = [img.size[::-1] for img in images]
        kwargs = dict(text_threshold=self.gd_text_threshold, target_sizes=target_sizes)
        try:
            # Newer transformers
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.gd_box_threshold, **kwargs
            )
        except TypeError:
            # Older transformers used `box_threshold`
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.gd_box_threshold, **kwargs
            )

        per_frame: list[list[dict]] = []
        for img, res in zip(images, results):
            rgb = img.convert("RGB")
            w, h = rgb.size
            instances: list[dict] = []
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()

            for (x0, y0, x1, y1), score in zip(boxes, scores):
                xmin, ymin = max(0, int(x0)), max(0, int(y0))
                xmax, ymax = min(w - 1, int(x1)), min(h - 1, int(y1))
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Note: this is box area, so it's larger than SAM 3's mask area
                if (xmax - xmin) * (ymax - ymin) / float(w * h) < self.min_mask_area_frac:
                    continue
                if self.drop_edge_touching and (xmin == 0 or ymin == 0 or xmax >= w - 1 or ymax >= h - 1):
                    continue

                pad_x = int((xmax - xmin) * self.crop_padding)
                pad_y = int((ymax - ymin) * self.crop_padding)
                box = (
                    max(0, xmin - pad_x),
                    max(0, ymin - pad_y),
                    min(w, xmax + pad_x + 1),
                    min(h, ymax + pad_y + 1),
                )
                instances.append(
                    {
                        "crop": rgb.crop(box),
                        "bbox": [xmin, ymin, xmax, ymax],
                        "score": float(score),
                    }
                )
            per_frame.append(instances)
        return per_frame

    def isolate_objects(self, images: list[Image.Image], text_prompt: str) -> list[list[dict]]:
        """Dispatch to whichever detector is configured."""
        if self.detector == "grounding_dino":
            return self.isolate_objects_with_grounding_dino(images, text_prompt)
        return self.isolate_objects_with_sam3(images, text_prompt)

    # ------------------------------------------------------------ resume state
    def _save_state(self, output_dir: Path, last_section: int) -> None:
        tmp = output_dir / "state.pt.tmp"
        torch.save(
            {
                "embedding_memory": self.embedding_memory,
                "object_counts": self.object_counts,
                "last_section": last_section,
            },
            tmp,
        )
        tmp.replace(output_dir / "state.pt")

    def _load_state(self, output_dir: Path) -> int:
        """Restore state from a previous run. Returns the next section index to process."""
        state_path = output_dir / "state.pt"
        if not state_path.exists():
            return 0
        state = torch.load(state_path, map_location="cpu")
        self.embedding_memory = state["embedding_memory"]
        self.object_counts = state["object_counts"]
        last = state["last_section"]
        for i in range(last + 1):
            p = output_dir / f"section_result_{i:04d}.json"
            if p.exists():
                with open(p) as f:
                    self.all_results.append(json.load(f))
        print(f"Resuming from section {last + 1} (counts so far: {self.object_counts})")
        return last + 1

    # --------------------------------------------------------------------- run
    def run(self) -> None:
        output_dir = Path(self.config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reuse FrameParser's section folders and sections.json, like the other pipelines.
        # They are created on first use if they don't exist yet.
        sections_dir = Path(self.frame_parser.output_dir)
        manifest_path = sections_dir / "sections.json"
        if not manifest_path.exists():
            self.frame_parser.create_section_dir()
        with open(manifest_path) as f:
            manifest = json.load(f)["sections"]
        section_names = sorted(manifest)

        section_paths = [list_frame_paths(sections_dir / name) for name in section_names]
        overlap_frac = self.config.get("section_overlap", 0.1)
        frame_size = self.config.get("max_frame_size", 640)

        first_section = self._load_state(output_dir)

        for section_idx, name in enumerate(section_names):
            if section_idx < first_section:
                continue

            # Overlap: also draw frames from the start of the next section
            paths = list(section_paths[section_idx])
            if section_idx + 1 < len(section_names):
                paths += section_paths[section_idx + 1][: int(len(paths) * overlap_frac)]

            start_s = float(manifest[name]["start_seconds"])
            end_s = float(manifest[name]["end_seconds"])
            time_str = f"{format_timestamp(start_s)} - {format_timestamp(end_s)}"

            pil_frames = sample_frames_from_paths(
                paths,
                num_samples=self.config.get("num_samples_per_section", 16),
                max_size=(frame_size, frame_size),
            )
            if not pil_frames:
                continue

            # 1. Ask the VLM for concepts, nudging it to reuse previously seen labels
            current_prompt = self.prompt
            if self.object_counts:
                known = ", ".join(self.object_counts.keys())
                current_prompt += (
                    f"\n\nCRITICAL: You previously detected these objects: [{known}]. "
                    "If any of these are still in the scene, you MUST reuse the exact same "
                    "label. Do not invent synonyms."
                )
            # Effective fps of the sampled frames (n frames spread over the section's duration)
            section_seconds = max(end_s - start_s, 1.0)
            sampled_fps = len(pil_frames) / section_seconds
            detection_response = self.ai_parser.call_vlm(
                current_prompt, folder_path=None, video=pil_frames, video_fps=sampled_fps
            )
            detected_concepts = self.parse_concept_list(detection_response)

            section_result = {
                "section": section_idx,
                "timestamp": time_str,
                "concepts_detected": detected_concepts,
                "rejected_concepts": [],  # VLM said it, SAM 3 found nothing (likely hallucination)
                "new_instances": {},      # label -> number counted as new in this section
                "max_in_single_frame": {},
                "is_unique_map": {},
            }

            # 2. Segment each concept, then dedup per frame against *earlier* memory only
            for label in detected_concepts:
                per_frame = self.isolate_objects(pil_frames, label)

                if not any(per_frame):
                    section_result["rejected_concepts"].append(label)
                    continue

                section_result["max_in_single_frame"][label] = max(len(f) for f in per_frame)
                new_in_section = 0

                for frame_idx, instances in enumerate(per_frame):
                    if not instances:
                        continue

                    embeddings = self.get_clip_embeddings([i["crop"] for i in instances])
                    flags = self.novelty_flags(embeddings, label)

                    new_embs = []
                    for inst_idx, (inst, emb, unique) in enumerate(zip(instances, embeddings, flags)):
                        section_result["is_unique_map"][f"{label}_f{frame_idx}_i{inst_idx}"] = unique
                        if not unique:
                            continue
                        new_embs.append(emb)
                        new_in_section += 1
                        self.object_counts[label] = self.object_counts.get(label, 0) + 1

                        if self.save_crops:
                            class_dir = output_dir / "crops" / label.replace(" ", "_")
                            class_dir.mkdir(parents=True, exist_ok=True)
                            inst["crop"].save(
                                class_dir / f"s{section_idx:04d}_f{frame_idx:02d}_i{inst_idx:02d}.jpg"
                            )

                    # Add to memory only after the whole frame is processed
                    if new_embs:
                        stacked = torch.stack(new_embs, dim=0)
                        prev = self.embedding_memory.get(label)
                        self.embedding_memory[label] = (
                            stacked if prev is None else torch.cat([prev, stacked], dim=0)
                        )

                section_result["new_instances"][label] = new_in_section

            self.write_json_output(output_dir, f"section_result_{section_idx:04d}.json", section_result)
            self.all_results.append(section_result)
            self._save_state(output_dir, section_idx)
            print(f"[{time_str}] section {section_idx}: counts so far {self.object_counts}")

        self.write_json_output(output_dir, "final_counts.json", self.object_counts)
        self.write_json_output(output_dir, "all_results.json", self.all_results)
        print(f"Pipeline execution finished. Final Object Counts: {self.object_counts}")