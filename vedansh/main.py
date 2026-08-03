import json
import os

# Force Hugging Face cache to scratch to prevent crashing your NCI home quota
os.environ["HF_HOME"] = "/scratch/pg06/vm4618/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"         # Force Hugging Face Hub offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # Force Transformers offline

from decord import VideoReader, cpu
from PIL import Image
from vllm import LLM, SamplingParams

def format_timestamp(seconds):
    """Converts seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def sample_frames_from_indices(vr, start_idx, end_idx, num_samples=8, max_size=(448, 448)):
    """Uniformly samples frames and resizes them to drastically cut visual token usage."""
    total_chunk_frames = end_idx - start_idx
    step = max(1, total_chunk_frames // num_samples)
    selected_indices = list(range(start_idx, end_idx, step))[:num_samples]
    
    batch = vr.get_batch(selected_indices).asnumpy()
    frames = []
    for frame in batch:
        img = Image.fromarray(frame)
        # Downscale image preserving aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        frames.append(img)
    return frames

def run_video_pipeline(
    video_path: str,
    full_history_json_path: str,
    change_analysis_json_path: str,
    vlm_model_name: str,
    frames_per_chunk: int = 300
):
    print("Loading modules...")
    # 1. Initialize Decord Video Reader
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    print(f"Loading VLM ({vlm_model_name}) into VRAM...")
    # 2. Initialize the single model for both Vision and Text tasks
    vlm = LLM(
        model=vlm_model_name, 
        enforce_eager=True, 
        dtype="half", 
        max_model_len=32768, 
        gpu_memory_utilization=0.85
    )

    vlm_params = SamplingParams(temperature=0.2, max_tokens=150)
    # Set temperature to 0.0 for strict rule adherence and deterministic JSON generation
    llm_params = SamplingParams(temperature=0.0, max_tokens=300)

    # Two distinct history buffers
    full_pipeline_history = []
    change_events_only = []
    
    previous_description = None
    chunk_idx = 0
    
    print("Starting video processing pipeline...")
    for start_frame in range(0, total_frames, frames_per_chunk):
        end_frame = min(start_frame + frames_per_chunk, total_frames)
        
        start_time_sec = start_frame / fps
        end_time_sec = end_frame / fps
        time_str = f"{format_timestamp(start_time_sec)} - {format_timestamp(end_time_sec)}"

        # Extract frames (Downscaled to 448x448 max, 8 samples per chunk)
        pil_frames = sample_frames_from_indices(vr, start_frame, end_frame, num_samples=8, max_size=(448, 448))

        # Step 1: Query VLM for scene description (Multi-Modal Prompt)
        vlm_prompt = "<|im_start|>user\n<|video_pad|>\nDescribe what is taking place in this scene in detail.<|im_end|>\n<|im_start|>assistant\n"
        vlm_output = vlm.generate(
            {"prompt": vlm_prompt, "multi_modal_data": {"video": pil_frames}},
            sampling_params=vlm_params
        )
        current_description = vlm_output[0].outputs[0].text.strip()

        # Step 2: Compare with Previous Scene using Few-Shot In-Context Learning
        change_detected = False
        change_record = None

        if previous_description is not None:
            comparison_prompt = f"""<|im_start|>system
You are an episodic memory assistant. Compare two consecutive video scene descriptions to identify ONLY major, interesting events or environment shifts.

CRITICAL THRESHOLD FOR INTERESTING EVENTS:
- SET `something_new: false` if the car is simply continuing to drive on the same road, adjusting lane position, or seeing similar trees/buildings. Most chunks should be false!
- SET `something_new: true` ONLY when a distinct, memorable event or landmark appears (e.g., entering a new city, passing a notable building like a dome, encountering a school bus, reaching a stop sign/intersection, sudden weather change).

STRICT QUESTION RULES:
- NEVER ask meta-questions like "What changed?", "What is the car's position?", or "What happens next?".
- NEVER use the words "scene", "previous", "current", or "change" in the question.
- Ask about factual details (e.g., landmarks, color of objects, number of lanes, roadside features) as if quizzing someone's memory of a driving trip.
- If `something_new: false`, set all other JSON fields to null.

--- EXAMPLE 1 (Mundane / Uninteresting -> IGNORE) ---
[PREVIOUS SCENE]: Car driving down a suburban street with trees on both sides.
[CURRENT SCENE]: Car continuing to drive down the same street, positioned in the center of the road.
JSON:
{{
  "something_new": false,
  "what_changed": null,
  "question": null,
  "answer": null
}}

--- EXAMPLE 2 (Interesting Milestone -> KEEP) ---
[PREVIOUS SCENE]: Suburban road with trees.
[CURRENT SCENE]: Entering a busy downtown city street with multi-lane traffic and a yellow school bus on the right.
JSON:
{{
  "something_new": true,
  "what_changed": "Transitioned from a quiet suburban road into a multi-lane city street with a school bus.",
  "question": "What specific yellow vehicle appeared on the right when entering the multi-lane city street?",
  "answer": "A yellow school bus."
}}

--- EXAMPLE 3 (Interesting Landmark -> KEEP) ---
[PREVIOUS SCENE]: Driving down a quiet road with a stop sign.
[CURRENT SCENE]: Driving towards a large dome-shaped industrial building in the distance.
JSON:
{{
  "something_new": true,
  "what_changed": "A large dome-shaped building appeared in the background.",
  "question": "What notable architectural structure came into view after passing the stop sign?",
  "answer": "A large dome-shaped building."
}}
<|im_end|>
<|im_start|>user
[PREVIOUS SCENE ({full_pipeline_history[-1]['timestamp']})]:
{previous_description}

[CURRENT SCENE ({time_str})]:
{current_description}
<|im_end|>
<|im_start|>assistant
"""
            # Generate purely using text via the same vlm object
            llm_output = vlm.generate(comparison_prompt, sampling_params=llm_params)
            comparison_raw = llm_output[0].outputs[0].text.strip()

            # Robust JSON extraction: Find the outer brackets {...} ignoring extra text
            try:
                json_start = comparison_raw.find("{")
                json_end = comparison_raw.rfind("}") + 1
                
                if json_start != -1 and json_end != 0:
                    clean_json = comparison_raw[json_start:json_end]
                    parsed = json.loads(clean_json)
                    
                    change_detected = parsed.get("something_new", False)

                    # Only record if something interesting actually changed AND a question exists
                    if change_detected and parsed.get("question"):
                        change_record = {
                            "chunk_index": chunk_idx,
                            "timestamp": time_str,
                            "start_seconds": start_time_sec,
                            "what_changed": parsed.get("what_changed"),
                            "generated_qa": {
                                "question": parsed.get("question"),
                                "answer": parsed.get("answer")
                            },
                            "previous_scene": previous_description,
                            "current_scene": current_description
                        }
                        change_events_only.append(change_record)
                else:
                    print(f"Warning: No valid JSON object found in chunk {chunk_idx} output.")

            except json.JSONDecodeError:
                print(f"JSON error at chunk {chunk_idx}. Raw output was:\n[{comparison_raw}]")

        # Step 3: Build Full Log Entry
        log_entry = {
            "chunk_index": chunk_idx,
            "timestamp": time_str,
            "start_seconds": start_time_sec,
            "vlm_description": current_description,
            "change_detected": change_detected
        }
        full_pipeline_history.append(log_entry)

        # Step 4: Stream save both files to actual filenames
        with open(full_history_json_path, "w", encoding="utf-8") as f:
            json.dump(full_pipeline_history, f, indent=2)

        with open(change_analysis_json_path, "w", encoding="utf-8") as f:
            json.dump(change_events_only, f, indent=2)

        print(f"Chunk {chunk_idx} ({time_str}) processed | Change Detected: {change_detected}")
        
        previous_description = current_description
        chunk_idx += 1

    return full_pipeline_history, change_events_only

if __name__ == "__main__":
    video_path = "/scratch/pg06/FYP2026S1_3473/boreas_dataset/boreas-2024-12-03-13-13/video.mp4"
    run_video_pipeline(
        video_path=video_path,
        full_history_json_path="full_history-boreas.json",
        change_analysis_json_path="change_analysis-boreas.json",
        vlm_model_name="/scratch/pg06/vm4618/huggingface_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    )