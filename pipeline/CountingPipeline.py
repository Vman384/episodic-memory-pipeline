import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import re

# Import SAM 3 and CLIP from Transformers
from transformers import CLIPProcessor, CLIPModel, Sam3Processor, Sam3Model

from pipeline.AIParser import AIParser
from pipeline.frame_parser import FrameParser, sample_frames_from_indices
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

    def isolate_objects_with_sam3(self, image: Image.Image, text_prompt: str) -> list:
        """Uses SAM 3 to find and mask ALL instances of a text concept in the image."""
        inputs = self.sam3_processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
            
        # Post-process to get binary masks
        target_sizes = [image.size[::-1]] # Format as (height, width)
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=target_sizes
        )[0]
        
        isolated_crops = []
        image_np = np.array(image.convert("RGB"))
        
        if "masks" in results:
            masks = results["masks"].cpu().numpy() # Shape: (N, H, W)
            for mask in masks:
                # Zero out pixels outside the predicted foreground mask
                isolated_image_np = image_np.copy()
                isolated_image_np[~mask] = 0
                isolated_crops.append(Image.fromarray(isolated_image_np))
                
        return isolated_crops

    def get_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extracts and L2-normalizes a CLIP visual embedding feature vector."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
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
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu()

    def is_unique(self, new_embedding: torch.Tensor, existing_embeddings: list) -> bool:
        """Checks if new embedding is distinct from past embeddings using cosine similarity."""
        if not existing_embeddings:
            return True

        memory_tensor = torch.cat(existing_embeddings, dim=0)
        similarities = F.cosine_similarity(new_embedding, memory_tensor, dim=1)
        max_similarity = similarities.max().item()

        return max_similarity <= self.similarity_threshold

    def parse_concept_list(self, vlm_response: str) -> list[str]:
        """Parses raw text/JSON from the VLM response into a list of string concepts."""
        if isinstance(vlm_response, list):
            return [str(item).strip() for item in vlm_response if item]

        # 1. Attempt JSON parsing if response is formatted as [ "red car", "person" ]
        try:
            # Locate bracketed JSON inside the string if surrounded by prose
            match = re.search(r"\[.*?\]", vlm_response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(x).strip().lower() for x in parsed if x]
        except json.JSONDecodeError:
            pass

        # 2. Fallback: Parse bullet points or comma-separated text
        lines = vlm_response.strip().split("\n")
        concepts = []
        for line in lines:
            # Clean out bullet points (*, -, 1., etc.)
            cleaned = re.sub(r"^[\s\*\-\d\.]+", "", line).strip().lower()
            if cleaned:
                concepts.extend([c.strip() for c in cleaned.split(",") if c.strip()])

        return list(set(concepts))

    def run(self) -> None:
        """Executes the pipeline using VLM for discovery and SAM 3 for spatial tracking."""
        sections_dir = Path(self.config["sections_dir"])
        sections = sorted(
            section for section in sections_dir.glob("section_*") if section.is_dir()
        )
        if sections:
            print(
                f"  Reusing {len(sections)} existing sections in {self.config['sections_dir']}"
            )
        else:
            # Create each section directory for segregated frames.
            sections = self.frame_parser.create_section_dir()
            print(f"  Created {len(sections)} sections in {self.config['sections_dir']}")
        output_dir = Path(self.config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for curr_section in sections:
            section_path = Path(curr_section)
            
            # 1. Ask VLM purely for a list of concepts (e.g. ["red car", "pedestrian"])
            detection_response = self.ai_parser.call_vlm(self.prompt, section_path)
            detected_concepts = self.parse_concept_list(detection_response)

            section_result = {
                "section": section_path.name,
                "concepts_detected": detected_concepts,
                "is_unique_map": {},
            }
            
            # Use the middle frame as our anchor for SAM 3
            frame_path = self.resolve_frame_path(section_path)
            base_image = Image.open(frame_path).convert("RGB")

            for concept in detected_concepts:
                # 2. SAM 3 dynamically searches for the text concept and returns masked crops
                isolated_crops = self.isolate_objects_with_sam3(base_image, concept)

                if not isolated_crops:
                    continue # SAM 3 didn't find anything matching the VLM's hallucination
                    
                if concept not in self.embedding_memory:
                    self.embedding_memory[concept] = []

                # 3. Process every instance of the concept SAM 3 found in this frame
                for idx, crop in enumerate(isolated_crops):
                    new_embedding = self.get_clip_embedding(crop)
                    unique = self.is_unique(new_embedding, self.embedding_memory[concept])
                    
                    instance_key = f"{concept}_{idx}"
                    section_result["is_unique_map"][instance_key] = unique

                    if unique:
                        self.object_counts[concept] = self.object_counts.get(concept, 0) + 1
                        self.embedding_memory[concept].append(new_embedding)

                        class_dir = output_dir / concept.replace(" ", "_")
                        class_dir.mkdir(parents=True, exist_ok=True)
                        
                        dest_name = f"{section_path.name}_inst_{idx}"
                        if section_path.is_dir():
                            shutil.copytree(section_path, class_dir / dest_name, dirs_exist_ok=True)
                        else:
                            shutil.copy(section_path, class_dir / f"{dest_name}_{section_path.name}")

            section_output_dir = output_dir / f"{section_path.name}_output"
            self.write_json_output(section_output_dir, "result.json", section_result)
            self.all_results.append(section_result)

        self.write_json_output(output_dir, "final_counts.json", self.object_counts)
        self.write_json_output(output_dir, "all_results.json", self.all_results)

        print(f"Pipeline execution finished. Final Object Counts: {self.object_counts}")