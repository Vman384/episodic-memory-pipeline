import json
import os
from pathlib import Path
import shutil
import numpy as np # type: ignore
from PIL import Image # type: ignore
import torch # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # type: ignore
import re
from decord import VideoReader, cpu
import time

# Import SAM 3 and CLIP from Transformers
from transformers import CLIPProcessor, CLIPModel, Sam3Processor, Sam3Model

from pipeline.AIParser import AIParser
from pipeline.frame_parser import FrameParser, format_timestamp, sample_frames_from_indices
from pipeline.ConfigLoader import CountingPipelineConfig

# Configure Hugging Face Cache Environment
os.environ["HF_HOME"] = "/scratch/pg06/vm4618/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class CountingPipeline:
    """Coordinates video frame sampling, VLM concept discovery, SAM 3 text-based segmentation, 
    and CLIP visual embedding deduplication for object counting."""

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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # State tracking
        self.embedding_memory = {}  # Dict[str, List[torch.Tensor]]
        self.object_counts = {}      # Dict[str, int]
        self.all_results = []
        
        # Tracking boxes across scenes to enforce label consistency
        self.last_scene_boxes = []  # List[Dict[str, any]] e.g. [{"label": "red car", "box": [x1, y1, x2, y2]}]
        
        # 1. Load CLIP Model & Processor
        clip_model_name = config.get("CLIPModelmodel", "openai/clip-vit-base-patch32")
        clip_proc_name = config.get("CLIPProcessormodel", "openai/clip-vit-base-patch32")
        
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_proc_name)
        self.clip_model.eval()

        # 2. Load SAM 3 Model & Processor
        sam3_model_name = config.get("sam3_model", "facebook/sam3")
        self.sam3_model = Sam3Model.from_pretrained(sam3_model_name).to(self.device)
        self.sam3_processor = Sam3Processor.from_pretrained(sam3_model_name)
        self.sam3_model.eval()

    def write_json_output(self, output_dir: Path, file_name: str, output: dict | list) -> None:
        """Utility to write output JSON files safely."""
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / file_name, "w") as result_file:
            json.dump(output, result_file, indent=2)

    def resolve_frame_path(self, section_dir: Path) -> Path:
        """Grabs the middle frame from the sampled video directory to run segmentation on."""
        section_path = Path(section_dir)
        frames = sorted(
            list(section_path.glob("*.jpg"))
            + list(section_path.glob("*.png"))
            + list(section_path.glob("*.jpeg"))
        )
        if not frames:
            raise FileNotFoundError(f"No image frames found in directory: {section_path}")
        
        return frames[len(frames) // 2]

    def get_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extracts and L2-normalizes a CLIP visual embedding feature vector."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device) # type: ignore
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            
            # Extract the raw tensor from the Hugging Face output object
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output # type: ignore
            elif hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds # type: ignore
            elif isinstance(image_features, tuple):
                image_features = image_features[0]
                
            # Now safe to apply PyTorch operations
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # type: ignore
        return image_features.cpu()

    def is_unique(self, new_embedding: torch.Tensor, existing_embeddings: list) -> bool:
        """Checks if new embedding is distinct from past embeddings using cosine similarity."""
        if not existing_embeddings:
            return True

        memory_tensor = torch.cat(existing_embeddings, dim=0)
        similarities = F.cosine_similarity(new_embedding, memory_tensor, dim=1)
        max_similarity = similarities.max().item()

        return max_similarity <= self.similarity_threshold

    def calculate_iou(self, boxA: list, boxB: list) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def parse_concept_list(self, vlm_response: str) -> list[str]:
        """Parses raw text/JSON from the VLM response into a list of string concepts."""
        if isinstance(vlm_response, list):
            return list(dict.fromkeys(
                str(item).strip().lower() for item in vlm_response if item
            ))

        cleaned_response = re.sub(
            r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", vlm_response
        ).strip()
        try:
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, list):
                return list(dict.fromkeys(
                    str(item).strip().lower() for item in parsed if item
                ))
        except json.JSONDecodeError:
            pass

        try:
            match = re.search(r"\[[\s\S]*?\]", cleaned_response)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return list(dict.fromkeys(
                        str(item).strip().lower() for item in parsed if item
                    ))
        except json.JSONDecodeError:
            pass
        # 2. Fallback: Parse bullet points or comma-separated text
        lines = cleaned_response.split("\n")
        concepts = []
        for line in lines:
            cleaned = re.sub(r"^[\s\*\-\d\.]+", "", line).strip().lower()
            if cleaned:
                concepts.extend([c.strip() for c in cleaned.split(",") if c.strip()])

        return list(set(concepts))

    def isolate_objects_with_sam3(self, images: list[Image.Image], text_prompt: str) -> list[tuple[Image.Image, list[int]]]:
        """Uses SAM 3 to isolate text concepts and returns BOTH the cropped image AND its bounding box."""
        prompts = [text_prompt] * len(images)
        
        inputs = self.sam3_processor(
            images=images, 
            text=prompts, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
            
        target_sizes = [img.size[::-1] for img in images]
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=target_sizes
        )
        
        isolated_crops_and_boxes = []
        for img, res in zip(images, results):
            if "masks" in res:
                image_np = np.array(img.convert("RGB"))
                masks = res["masks"].cpu().numpy()
                
                for mask in masks:
                    # Extract bounding box from the boolean mask
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue # Mask is empty
                        
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
                    
                    # Create the blacked-out crop
                    isolated_image_np = image_np.copy()
                    isolated_image_np[~mask] = 0
                    crop_img = Image.fromarray(isolated_image_np)
                    
                    isolated_crops_and_boxes.append((crop_img, bbox))
                    
        return isolated_crops_and_boxes

    def run(self) -> None:
        """Executes the pipeline using VLM for discovery and SAM 3 for spatial tracking."""
        vr = VideoReader(self.config["frames_dir"], ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr) - self.config["step"] - 5

        output_dir = Path(self.config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for start_frame in range(0, total_frames, self.config["frames_per_section"]):
            end_frame = min(start_frame + self.config["frames_per_section"], total_frames)
            
            start_time_sec = start_frame / fps
            end_time_sec = end_frame / fps
            time_str = f"{format_timestamp(start_time_sec)} - {format_timestamp(end_time_sec)}"

            frame_size = self.config.get("max_frame_size", 640)
            pil_frames = sample_frames_from_indices(
                vr,
                start_frame,
                end_frame,
                num_samples=self.config.get("num_samples_per_section", 16),
                max_size=(frame_size, frame_size),
            )
            
            # 1. DYNAMIC PROMPTING: Force the VLM to reuse known labels
            current_prompt = self.prompt
            if self.object_counts:
                known_labels = ", ".join(self.object_counts.keys())
                current_prompt += f"\n\nCRITICAL: You previously detected these objects: [{known_labels}]. If any of these are still in the scene, you MUST reuse the exact same label. Do not invent synonyms."

            # 2. Ask VLM purely for a list of concepts
            detection_response = self.ai_parser.call_vlm(current_prompt, folder_path=None, video=pil_frames)
            detected_concepts = self.parse_concept_list(detection_response)

            section_result = {
                "concepts_detected": detected_concepts,
                "is_unique_map": {},
                "timestamp": time_str
            }
            
            current_scene_boxes = []

            for concept in detected_concepts:
                # 3. SAM 3 returns both the masked crops AND the bounding boxes
                crops_and_boxes = self.isolate_objects_with_sam3(pil_frames, concept)

                if not crops_and_boxes:
                    continue 
                    
                # 4. Process every instance of the concept
                for idx, (crop, bbox) in enumerate(crops_and_boxes):
                    
                    # SPATIAL IOU TRACKING: Check if this box overlaps with an object from the previous scene
                    assigned_label = concept
                    best_iou = 0.0
                    
                    for past_item in self.last_scene_boxes:
                        iou = self.calculate_iou(bbox, past_item["box"])
                        if iou > best_iou:
                            best_iou = iou
                            assigned_label = past_item["label"] # Inherit the old label
                            
                    if best_iou > 0.50:
                        print(f"[{time_str}] Tracking match: VLM said '{concept}', mapped to '{assigned_label}' (IoU: {best_iou:.2f})")
                    
                    # Track this box for the next scene
                    current_scene_boxes.append({"label": assigned_label, "box": bbox})

                    # Ensure we have memory initialized for the potentially updated label
                    if assigned_label not in self.embedding_memory:
                        self.embedding_memory[assigned_label] = []

                    # 5. CLIP Deduplication using the tracking-adjusted label
                    new_embedding = self.get_clip_embedding(crop)
                    unique = self.is_unique(new_embedding, self.embedding_memory[assigned_label])
                    
                    instance_key = f"{assigned_label}_{idx}"
                    section_result["is_unique_map"][instance_key] = unique

                    if unique:
                        self.object_counts[assigned_label] = self.object_counts.get(assigned_label, 0) + 1
                        self.embedding_memory[assigned_label].append(new_embedding)

                        class_dir = output_dir / assigned_label.replace(" ", "_")
                        class_dir.mkdir(parents=True, exist_ok=True)

            # Update the global tracker to pass these boxes to the next chunk
            self.last_scene_boxes = current_scene_boxes

            self.write_json_output(output_dir, f"result_{time.time()}.json", section_result)
            self.all_results.append(section_result)

        self.write_json_output(output_dir, f"final_counts_{time.time()}.json", self.object_counts)
        self.write_json_output(output_dir, f"all_results_{time.time()}.json", self.all_results)

        print(f"Pipeline execution finished. Final Object Counts: {self.object_counts}")