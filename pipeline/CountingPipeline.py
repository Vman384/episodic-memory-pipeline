"""Sparse event localisation pipeline."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import os

# Import SAM 2 Predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pipeline.AIParser import AIParser
from pipeline.frame_parser import FrameParser
from pipeline.frame_parser import sample_frames_from_indices
from pipeline.ConfigLoader import CountingPipelineConfig


os.environ["HF_HOME"] = "/scratch/pg06/vm4618/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"         # Force Hugging Face Hub offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # Force Transformers offline
class CountingPipeline:
    """Coordinate configuration, frame parsing, and sparse-event VLM calls."""

    def __init__(
        self,
        config: CountingPipelineConfig,
        prompt: str,
        frame_parser: FrameParser,
        ai_parser: AIParser,
    ):
        self.prompt = prompt
        self.config = config
        self.frame_parser = frame_parser
        self.ai_parser = ai_parser
        self.embedding_memory = {}
        
        # Load CLIP
        self.model = CLIPModel.from_pretrained(config["CLIPModelmodel"])
        self.processor = CLIPProcessor.from_pretrained(config["CLIPProcessormodel"])
        
        # Load SAM 2
        sam2_checkpoint = config.get("sam2_checkpoint", "sam2_hiera_large.pt")
        model_cfg = config.get("sam2_model_cfg", "sam2_hiera_l.yaml")
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        self.sam_predictor = SAM2ImagePredictor(sam2_model)

        self.all_results = []
        self.object_counts = {}

    def write_json_output(self, output_dir: Path, file_name: str, output: dict) -> None:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / file_name, "w") as result_file:
            json.dump(output, result_file, indent=2)
    
    def isolate_object_with_sam(self, image_path: str, bbox: list) -> Image.Image:
        """Uses SAM 2 to mask out the background based on a bounding box."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 1. Load image into SAM 2
        self.sam_predictor.set_image(image_np)
        
        # 2. Predict the mask using the VLM's bounding box [x_min, y_min, x_max, y_max]
        input_box = np.array(bbox)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False, # We only want the best mask
        )
        
        # 3. Apply the mask to black out the background
        mask = masks[0]
        isolated_image_np = image_np.copy()
        isolated_image_np[~mask] = 0 # Set non-object pixels to black
        
        return Image.fromarray(isolated_image_np)

    def get_clip_embedding(self, image: Image.Image):
        """Converts a PIL Image into a normalized 512-dimensional vector."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def is_unique(self, new_embedding, existing_embeddings, threshold=0.85):
        if not existing_embeddings:
            return True
        memory_tensor = torch.cat(existing_embeddings)
        similarities = F.cosine_similarity(new_embedding, memory_tensor, dim=1)
        max_similarity = similarities.max().item()
        
        if max_similarity > threshold:
            return False
        return True

    def run(self) -> None:
        sections = sample_frames_from_indices(
            self.config["frames_dir"],
            self.config["sampled_frames_dir"],
            self.config["sampled_indices"],
        )

        output_dir = Path(self.config["output"])

        for curr_section in sections:
            # 1. Ask your VLM what is in the scene AND where it is
            detection_response = self.ai_parser.call_vlm(self.prompt, curr_section)
            
            # EXPECTED FORMAT: [("red car", [100, 150, 400, 350]), ...]
            detected_objects = self.ai_parser.parse_objects_with_boxes(detection_response) 

            section_result = {
                "section": curr_section.name,
                "objects_detected": [obj[0] for obj in detected_objects],
                "is_unique_map": {}
            }

            for obj_class, bbox in detected_objects:
                frame_path = self.extract_middle_frame(curr_section) 
                
                # 2. SAM 2 intercepts the frame and isolates the object
                isolated_image = self.isolate_object_with_sam(frame_path, bbox)
                
                # 3. CLIP generates a clean embedding without background noise
                new_embedding = self.get_clip_embedding(isolated_image)
                
                if obj_class not in self.embedding_memory:
                    self.embedding_memory[obj_class] = []
                    
                unique = self.is_unique(new_embedding, self.embedding_memory[obj_class])
                section_result["is_unique_map"][obj_class] = unique

                if unique:
                    self.object_counts[obj_class] = self.object_counts.get(obj_class, 0) + 1
                    self.embedding_memory[obj_class].append(new_embedding)
                    
                class_dir = output_dir / obj_class
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(curr_section, class_dir)
                
            section_output_dir = output_dir / f"{curr_section.name}_output"
            self.write_json_output(section_output_dir, "result.json", section_result)
            self.all_results.append(section_result)

        self.write_json_output(output_dir, "all_results.json", self.all_results)
        print(f"Final Counts: {self.object_counts}")