import json
import os
from decord import VideoReader, cpu
from PIL import Image
from vllm import LLM, SamplingParams

def format_timestamp(seconds):
    """Converts seconds into MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def sample_frames_from_indices(vr, start_idx, end_idx, num_samples=16):
    """Uniformly samples 'num_samples' frames between start and end indices."""
    total_chunk_frames = end_idx - start_idx
    step = max(1, total_chunk_frames // num_samples)
    selected_indices = list(range(start_idx, end_idx, step))[:num_samples]
    
    batch = vr.get_batch(selected_indices).asnumpy()
    return [Image.fromarray(frame) for frame in batch]

def run_video_pipeline(
    video_path: str,
    full_history_json_path: str,
    change_analysis_json_path: str,
    vlm_model_name: str,
    text_llm_model_name: str,
    frames_per_chunk: int = 300
):
    # 1. Initialize Decord Video Reader
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # 2. Initialize VLM and LLM
    vlm = LLM(
        model=vlm_model_name, 
        enforce_eager=True, 
        dtype="half", 
        max_model_len=4096, 
        gpu_memory_utilization=0.45
    )
    
    text_llm = LLM(
        model=text_llm_model_name, 
        dtype="half", 
        max_model_len=4096, 
        gpu_memory_utilization=0.45
    )

    vlm_params = SamplingParams(temperature=0.2, max_tokens=150)
    llm_params = SamplingParams(temperature=0.3, max_tokens=300)

    # Two distinct history buffers
    full_pipeline_history = []
    change_events_only = []
    
    previous_description = None
    chunk_idx = 0
    
    for start_frame in range(0, total_frames, frames_per_chunk):
        end_frame = min(start_frame + frames_per_chunk, total_frames)
        
        start_time_sec = start_frame / fps
        end_time_sec = end_frame / fps
        time_str = f"{format_timestamp(start_time_sec)} - {format_timestamp(end_time_sec)}"

        # Extract frames
        pil_frames = sample_frames_from_indices(vr, start_frame, end_frame, num_samples=16)

        # Step 1: Query VLM for scene description
        vlm_prompt = "<|im_start|>user\n<|video_pad|>\nDescribe what is taking place in this scene in detail.<|im_end|>\n<|im_start|>assistant\n"
        vlm_output = vlm.generate(
            {"prompt": vlm_prompt, "multi_modal_data": {"video": pil_frames}},
            sampling_params=vlm_params
        )
        current_description = vlm_output[0].outputs[0].text.strip()

        # Step 2: Compare with Previous Scene using Text LLM
        change_detected = False
        change_record = None

        if previous_description is not None:
            comparison_prompt = f"""<|im_start|>system
You are a video analysis assistant. Compare two consecutive video scene descriptions and identify if any NEW significant action, subject, or state change occurred.
Return ONLY a valid JSON object matching this schema:
{{
  "something_new": true/false,
  "what_changed": "Description of change, or null if nothing changed",
  "question": "A question based on the new scene, or null if nothing changed",
  "answer": "The answer to the question based on the new scene, or null if nothing changed"
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
            llm_output = text_llm.generate(comparison_prompt, sampling_params=llm_params)
            comparison_raw = llm_output[0].outputs[0].text.strip()

            try:
                parsed = json.loads(comparison_raw)
                change_detected = parsed.get("something_new", False)

                if change_detected:
                    # Construct dedicated change object
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

            except json.JSONDecodeError:
                print(f"JSON error, llm output [{llm_output}]")
                pass

        # Step 3: Build Full Log Entry
        log_entry = {
            "chunk_index": chunk_idx,
            "timestamp": time_str,
            "start_seconds": start_time_sec,
            "vlm_description": current_description,
            "change_detected": change_detected
        }
        full_pipeline_history.append(log_entry)

        # Step 4: Stream save both files to disk after every chunk iteration
        with open(full_history_json_path, "w", encoding="utf-8") as f:
            json.dump(full_pipeline_history, f, indent=2)

        with open(change_analysis_json_path, "w", encoding="utf-8") as f:
            json.dump(change_events_only, f, indent=2)

        print(f"Chunk {chunk_idx} ({time_str}) processed | Change Detected: {change_detected}")
        
        previous_description = current_description
        chunk_idx += 1

    return full_pipeline_history, change_events_only

run_video_pipeline("test.mp4","./","./","Qwen/Qwen2-VL-7B-Instruct","deepseek-ai/DeepSeek-V4-Flash")