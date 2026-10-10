"""Send either sampled frames or a native MP4 input to a configured VLM."""

import math
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.AIParser import AIParser
from pipeline.timeframe_converter import parse_frame_timestamp


class VideoAIParser:
    """Video input adapter selected by ``input_mode`` in the model config."""

    INPUT_MODES = {"frames", "native_video"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, config: dict, input_mode: str | None = None):
        configured_mode = input_mode or config.get("input_mode", "frames")
        if not isinstance(configured_mode, str):
            raise ValueError("input_mode must be 'frames' or 'native_video'")

        self.input_mode = configured_mode.lower()
        self.backend = config.get("backend", "local").lower()
        self.config = config

        if self.input_mode not in self.INPUT_MODES:
            raise ValueError("input_mode must be 'frames' or 'native_video'")
        if self.input_mode == "frames":
            if self.backend not in {"local", "api"}:
                raise ValueError("Frame input requires backend 'local' or 'api'")
            self.ai_parser = AIParser(config)
        elif self.backend == "local":
            model_name = str(config.get("model", "")).lower()
            if "qwen3-vl" not in model_name:
                raise ValueError("Native local video input currently supports Qwen3-VL models")
            os.environ.setdefault(
                "MODEL_SEQ_LEN",
                str(config.get("max_model_len", 128000)),
            )
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as error:
                raise RuntimeError(
                    "Native Qwen3-VL video input requires qwen-vl-utils; "
                    "install the project requirements"
                ) from error
            self.process_vision_info = process_vision_info
            self.ai_parser = AIParser(config)
        elif self.backend == "gemini":
            self.ai_parser = None
        else:
            raise ValueError(
                "Native MP4 input requires backend 'local' for Qwen3-VL or "
                "'gemini' for the Gemini API"
            )

    def _call_local_native_video(self, video_path: Path, prompt: str) -> tuple[str, dict]:
        video_part = {
            "type": "video",
            "video": video_path.resolve().as_uri(),
        }
        if self.config.get("video_fps") is not None:
            video_part["fps"] = self.config["video_fps"]
        if self.config.get("video_max_frames") is not None:
            video_part["max_frames"] = self.config["video_max_frames"]

        messages = [
            {
                "role": "user",
                "content": [video_part, {"type": "text", "text": prompt}],
            }
        ]
        processor = self.ai_parser.processor
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        multimodal_data = {}
        if image_inputs is not None:
            multimodal_data["image"] = image_inputs
        if video_inputs is not None:
            multimodal_data["video"] = video_inputs
        if "video" not in multimodal_data:
            raise ValueError(f"Could not decode video input: {video_path}")

        llm_input = {
            "prompt": formatted_prompt,
            "multi_modal_data": multimodal_data,
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = self.ai_parser.model.generate(
            llm_input,
            sampling_params=self.ai_parser.sampling_params,
        )
        metadata = {
            "provider": "vllm",
            "video_size_bytes": video_path.stat().st_size,
            "video_fps": self.config.get("video_fps", 2.0),
            "video_max_frames": self.config.get("video_max_frames", 768),
            "decoder": "qwen-vl-utils",
        }
        return self.ai_parser._get_text(outputs), metadata

    def _call_native_video(self, video_path: Path, prompt: str) -> tuple[str, dict]:
        try:
            from dotenv import load_dotenv
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Native Gemini video input requires google-genai and python-dotenv; "
                "install the project requirements"
            ) from error

        load_dotenv()
        api_key_env = self.config.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini backend requires an API key in environment variable {api_key_env}"
            )

        client = genai.Client(api_key=api_key)
        uploaded_file = client.files.upload(file=str(video_path))
        uploaded_file_name = uploaded_file.name
        timeout_seconds = float(
            self.config.get("video_processing_timeout_seconds", 1800)
        )
        deadline = time.monotonic() + timeout_seconds

        try:
            while True:
                state = getattr(uploaded_file, "state", None)
                state_name = getattr(state, "name", str(state)).upper().rsplit(".", 1)[-1]
                if state_name == "ACTIVE":
                    break
                if state_name in {"FAILED", "ERROR"}:
                    raise RuntimeError(
                        f"Gemini could not process uploaded video: {state_name}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for Gemini to process the uploaded video"
                    )
                time.sleep(5)
                uploaded_file = client.files.get(name=uploaded_file_name)

            generation_config = {}
            max_output_tokens = self.config.get("max_tokens", 4000)
            if max_output_tokens is not None:
                generation_config["max_output_tokens"] = max_output_tokens
            temperature = self.config.get("api_temperature")
            if temperature is None:
                temperature = self.config.get("temperature")
            if temperature is not None:
                generation_config["temperature"] = temperature

            response = client.models.generate_content(
                model=self.config["model"],
                contents=[uploaded_file, prompt],
                config=(
                    types.GenerateContentConfig(**generation_config)
                    if generation_config
                    else None
                ),
            )
            response_text = response.text or ""
            metadata = {
                "provider": "gemini",
                "video_size_bytes": video_path.stat().st_size,
                "uploaded_file_state": state_name,
            }
            return response_text, metadata
        finally:
            try:
                client.files.delete(name=uploaded_file_name)
            except Exception as error:
                print(
                    f"Could not delete temporary Gemini upload {uploaded_file_name}: {error}",
                    flush=True,
                )

    @staticmethod
    def _write_sampled_frames(
        video_path: Path,
        output_dir: Path,
        max_frames: int,
        max_image_size: int,
    ) -> tuple[list[Path], dict]:
        if max_frames < 0:
            raise ValueError("max_frames must be 0 (all frames) or greater")
        if max_image_size < 1:
            raise ValueError("max_image_size must be at least 1")
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            from decord import VideoReader, cpu
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Video input requires decord and Pillow; install the project requirements"
            ) from error

        reader = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(reader)
        if total_frames == 0:
            raise ValueError(f"Video contains no frames: {video_path}")

        fps = float(reader.get_avg_fps())
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not read a valid frame rate from: {video_path}")

        frame_count = total_frames if max_frames == 0 else min(total_frames, max_frames)
        if frame_count == 1:
            frame_indices = [round((total_frames - 1) / 2)]
        else:
            last_index = total_frames - 1
            frame_indices = sorted(
                {round(index * last_index / (frame_count - 1)) for index in range(frame_count)}
            )

        frame_paths = []
        timestamps = []
        for sample_number, frame_index in enumerate(frame_indices, start=1):
            timestamp = frame_index / fps
            image = Image.fromarray(reader[frame_index].asnumpy()).convert("RGB")
            image.thumbnail(
                (max_image_size, max_image_size),
                Image.Resampling.LANCZOS,
            )
            frame_path = output_dir / f"frame_{sample_number:04d}.jpg"
            image.save(frame_path, format="JPEG", quality=85)
            frame_paths.append(frame_path)
            timestamps.append(round(timestamp, 3))

        metadata = {
            "total_frames": total_frames,
            "fps": fps,
            "frames_used": len(frame_paths),
            "sampled_frame_indices": frame_indices,
            "sampled_timestamps_seconds": timestamps,
        }
        return frame_paths, metadata

    @staticmethod
    def _write_sampled_frame_directory(
        frames_dir: Path,
        output_dir: Path,
        max_frames: int,
        max_image_size: int,
    ) -> tuple[list[Path], dict]:
        if max_frames < 0:
            raise ValueError("max_frames must be 0 (all frames) or greater")
        if max_image_size < 1:
            raise ValueError("max_image_size must be at least 1")
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Frame input requires Pillow; install the project requirements"
            ) from error

        frames = [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VideoAIParser.IMAGE_EXTENSIONS
        ]
        frames.sort(key=parse_frame_timestamp)
        if not frames:
            raise ValueError(f"No image frames found in folder: {frames_dir}")

        frame_count = len(frames) if max_frames == 0 else min(len(frames), max_frames)
        if frame_count == 1:
            selected_indices = [round((len(frames) - 1) / 2)]
        else:
            last_index = len(frames) - 1
            selected_indices = sorted(
                {
                    round(index * last_index / (frame_count - 1))
                    for index in range(frame_count)
                }
            )

        sampled_paths = []
        for sample_number, index in enumerate(selected_indices, start=1):
            with Image.open(frames[index]) as source_image:
                image = source_image.convert("RGB")
            image.thumbnail(
                (max_image_size, max_image_size),
                Image.Resampling.LANCZOS,
            )
            sampled_path = output_dir / f"frame_{sample_number:04d}.jpg"
            image.save(sampled_path, format="JPEG", quality=85)
            sampled_paths.append(sampled_path)

        metadata = {
            "source": "frames_dir",
            "frames_dir": str(frames_dir),
            "total_frames": len(frames),
            "frames_used": len(sampled_paths),
            "sampled_frame_indices": selected_indices,
            "sampled_frame_names": [frames[index].name for index in selected_indices],
        }
        return sampled_paths, metadata

    def _call_frame_images(
        self,
        prompt: str,
        frame_paths: list[Path],
        metadata: dict,
        api_batch_size: int,
    ) -> tuple[str, dict]:
        if self.backend != "api":
            response = self.ai_parser.call_vlm_images(
                prompt,
                frame_paths,
                include_frame_manifest=False,
            )
            return response, {**metadata, "provider": "vllm"}

        if api_batch_size < 1:
            raise ValueError("api_batch_size must be at least 1")

        previous_response_id = None
        frames_received = 0
        batches = []
        total_frames = len(frame_paths)
        batch_count = (total_frames + api_batch_size - 1) // api_batch_size

        for batch_number, start in enumerate(
            range(0, total_frames, api_batch_size),
            start=1,
        ):
            batch = frame_paths[start : start + api_batch_size]
            first_frame = start + 1
            last_frame = start + len(batch)
            batch_prompt = (
                f"Frame batch {batch_number} of {batch_count}: frames "
                f"{first_frame}-{last_frame} of {total_frames}, in chronological "
                "order from the drive. Remember these visual details for the "
                "questions after all batches. Reply with OK only."
            )
            try:
                response = self.ai_parser.call_api_conversation(
                    self.ai_parser.build_api_message(batch_prompt, batch),
                    previous_response_id=previous_response_id,
                )
            except RuntimeError as error:
                print(
                    f"Stopped before sampled frame {first_frame}: {error}",
                    flush=True,
                )
                batches.append(
                    {
                        "frames_from": first_frame,
                        "frames_to": last_frame,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                break

            previous_response_id = response.id
            frames_received = last_frame
            usage = getattr(response, "usage", None)
            batches.append(
                {
                    "frames_from": first_frame,
                    "frames_to": last_frame,
                    "status": "ok",
                    "input_tokens": getattr(usage, "input_tokens", None),
                }
            )
            print(
                f"Sampled frames {first_frame}-{last_frame} accepted "
                f"({batch_number}/{batch_count})",
                flush=True,
            )

        sampled_frames = metadata.get("frames_used", total_frames)
        metadata = {
            **metadata,
            "provider": "api",
            "api_batch_size": api_batch_size,
            "frames_sampled": sampled_frames,
            "frames_received": frames_received,
            "frames_used": frames_received,
            "api_batches": batches,
        }
        if previous_response_id is None:
            metadata["error"] = "No frame batches were accepted by the API"
            return "", metadata

        try:
            response = self.ai_parser.call_api_conversation(
                prompt,
                previous_response_id=previous_response_id,
            )
        except RuntimeError as error:
            metadata["question_error"] = str(error)
            return "", metadata
        return response.output_text, metadata

    def call_frames_dir(
        self,
        frames_dir: str | Path,
        prompt: str,
        max_frames: int = 100,
        max_image_size: int = 768,
        api_batch_size: int = 5,
    ) -> tuple[str, dict]:
        """Ask a frame-input model about uniformly sampled images in a folder."""
        if self.input_mode != "frames":
            raise ValueError("--frames-dir can only be used with input_mode 'frames'")

        frames_dir = Path(frames_dir).expanduser()
        with TemporaryDirectory(prefix="qa_camera_frames_") as temporary_directory:
            frame_paths, metadata = self._write_sampled_frame_directory(
                frames_dir,
                Path(temporary_directory),
                max_frames,
                max_image_size,
            )
            response, metadata = self._call_frame_images(
                prompt,
                frame_paths,
                metadata,
                api_batch_size,
            )
        return response, metadata

    def call_video(
        self,
        video_path: str | Path,
        prompt: str,
        max_frames: int = 100,
        max_image_size: int = 768,
        api_batch_size: int = 5,
    ) -> tuple[str, dict]:
        """Ask the model about a source video using its configured input mode."""
        video_path = Path(video_path).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if self.input_mode == "native_video":
            if video_path.suffix.lower() != ".mp4":
                raise ValueError("Native video input currently expects an .mp4 file")
            if self.backend == "local":
                return self._call_local_native_video(video_path, prompt)
            return self._call_native_video(video_path, prompt)

        if max_frames < 0:
            raise ValueError("max_frames must be 0 (all frames) or greater")
        if max_image_size < 1:
            raise ValueError("max_image_size must be at least 1")

        with TemporaryDirectory(prefix="qa_video_frames_") as temporary_directory:
            frame_paths, metadata = self._write_sampled_frames(
                video_path,
                Path(temporary_directory),
                max_frames,
                max_image_size,
            )
            response, metadata = self._call_frame_images(
                prompt,
                frame_paths,
                metadata,
                api_batch_size,
            )
        return response, metadata
