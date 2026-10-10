import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

try:
    from pipeline.timeframe_converter import parse_frame_timestamp
except ImportError:
    from timeframe_converter import parse_frame_timestamp


class AIParser:
    """
    Interface for local vLLM and OpenAI-compatible API inference.
    """

    API_RETRY_INITIAL_DELAY_SECONDS = 2
    API_RETRY_MAX_DELAY_SECONDS = 300

    IMAGE_MEDIA_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, config: dict):
        backend = config.get("backend", "local").lower()
        model = config.get("model")
        temperature = config.get("temperature")
        max_tokens = config.get("max_tokens", 100)

        if backend not in {"local", "api"}:
            raise ValueError("backend must be either 'local' or 'api'")
        if not model:
            raise ValueError("A model must be configured")

        self.backend = backend
        self.model_name = model
        self.local_temperature = 0.2 if temperature is None else temperature
        self.api_temperature = config.get("api_temperature")
        self.max_tokens = max_tokens
        self.reasoning_effort = config.get("reasoning_effort")
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if backend == "local":
            # Set offline mode before importing libraries that may resolve models.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            llm_config = {
                "model": model,
                "enforce_eager": config.get("enforce_eager", True),
                "dtype": config.get("dtype", "half"),
                "max_model_len": config.get("max_model_len", 4096),
                "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.9),
                "tensor_parallel_size": config.get("tensor_parallel_size", 1),
            }
            if config.get("limit_mm_per_prompt") is not None:
                llm_config["limit_mm_per_prompt"] = config["limit_mm_per_prompt"]

            self.model = LLM(
                **llm_config,
            )
            self.sampling_params = SamplingParams(
                temperature=self.local_temperature,
                max_tokens=max_tokens,
            )
            # Load tokenizer/processor to generate chat templates and image markers
            self.processor = AutoProcessor.from_pretrained(
                model,
                local_files_only=True,
            )

        elif backend == "api":
            from dotenv import load_dotenv
            from openai import APIConnectionError, OpenAI

            load_dotenv()
            api_key = os.getenv(config.get("api_key_env"))
            if not api_key:
                raise ValueError("API mode requires an API key")

            api_base_url = config.get("api_base_url")
            if not api_base_url:
                raise ValueError("API Base URL not provided!")

            # api_base_url = api_base_url.rstrip("/").removesuffix("/responses")

            # OpenCode Go uses this stable per-run ID for routing and prompt
            # caching. Generic clients without it can be rejected or throttled.
            self.api_session_id = f"ses_{uuid.uuid4().hex}"
            self.api_connection_error = APIConnectionError
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
                default_headers={"x-opencode-session": self.api_session_id},
                # Transient failures are retried in _call_api with backoff,
                # without the SDK's separate fixed retry limit.
                max_retries=0,
            )
            self.api_base_url = api_base_url

    @staticmethod
    def _get_text(outputs) -> str:
        """Return the first generated response as plain text."""
        return outputs[0].outputs[0].text

    @staticmethod
    def _get_api_text(response) -> str:
        """Return text from an OpenAI Responses API response."""
        return response.output_text

    @staticmethod
    def _api_status_and_headers(error):
        response = getattr(error, "response", None)
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        headers = getattr(response, "headers", {}) or {}
        return status_code, headers

    def _is_retryable_api_error(self, error) -> bool:
        status_code, headers = self._api_status_and_headers(error)

        # Respect an explicit retry instruction from the API gateway.
        should_retry = str(headers.get("x-should-retry", "")).lower()
        if should_retry == "true":
            return True
        if should_retry == "false":
            return False

        if status_code is not None:
            try:
                status_code = int(status_code)
            except (TypeError, ValueError):
                return False

            body = getattr(error, "body", None)
            api_error = body.get("error", {}) if isinstance(body, dict) else {}
            if status_code == 429 and isinstance(api_error, dict):
                if (
                    api_error.get("code") == "insufficient_quota"
                    or api_error.get("type") == "insufficient_quota"
                ):
                    return False
            return status_code in {408, 409, 429} or status_code >= 500

        # Connection errors include network failures and request timeouts.
        return isinstance(error, self.api_connection_error)

    @staticmethod
    def _retry_after_seconds(headers) -> float | None:
        retry_after_ms = headers.get("retry-after-ms")
        if retry_after_ms is not None:
            try:
                return max(0.0, float(retry_after_ms) / 1000)
            except (TypeError, ValueError):
                pass

        retry_after = headers.get("retry-after")
        if retry_after is None:
            return None

        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            # Retry-After may also be an HTTP date rather than a number of seconds.
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

    def _api_retry_delay(self, error, retry_number: int) -> float:
        # Exponential backoff with jitter, capped at five minutes.
        exponent = min(retry_number - 1, 8)
        delay_cap = min(
            self.API_RETRY_MAX_DELAY_SECONDS,
            self.API_RETRY_INITIAL_DELAY_SECONDS * (2**exponent),
        )
        delay = random.uniform(delay_cap / 2, delay_cap)

        _, headers = self._api_status_and_headers(error)
        retry_after = self._retry_after_seconds(headers)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    def _format_api_error(self, error) -> str:
        status_code, headers = self._api_status_and_headers(error)
        details = getattr(error, "body", None)
        if not details:
            response = getattr(error, "response", None)
            details = getattr(response, "text", None)
        if isinstance(details, (dict, list)):
            details = json.dumps(details)
        details = details or str(error)
        request_id = getattr(error, "request_id", None)
        if not request_id:
            request_id = headers.get("x-request-id")
        request_info = f" request_id={request_id}" if request_id else ""
        return (
            f"API request failed for model {self.model_name} at "
            f"{self.api_base_url} (status={status_code}{request_info}): {details}"
        )

    def _build_api_request(
        self,
        input_data: str | list[dict],
        max_tokens: int | None = None,
        conversation: bool = False,
        previous_response_id: str | None = None,
    ) -> dict:
        request = {
            "model": self.model_name,
            "input": input_data,
            "max_output_tokens": max_tokens
            if max_tokens is not None
            else self.max_tokens,
        }

        if self.api_temperature is not None:
            request["temperature"] = self.api_temperature
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if conversation:
            # Stored responses let the next request continue this conversation.
            # Truncation is disabled so an over-full context fails explicitly
            # instead of silently dropping earlier input.
            request["store"] = True
            request["truncation"] = "disabled"
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        return request

    def _create_api_response(self, request: dict):
        retry_number = 0
        while True:
            try:
                return self.client.responses.create(**request)
            except Exception as error:
                if not self._is_retryable_api_error(error):
                    raise RuntimeError(self._format_api_error(error)) from error

                retry_number += 1
                delay = self._api_retry_delay(error, retry_number)
                print(
                    f"{self._format_api_error(error)}; transient failure, "
                    f"retrying in {delay:.1f}s (retry {retry_number})",
                    flush=True,
                )
                time.sleep(delay)

    def _call_api(self, input_data: str | list[dict], max_tokens: int | None = None) -> str:
        request = self._build_api_request(input_data, max_tokens)
        return self._get_api_text(self._create_api_response(request))

    def call_api_conversation(
        self,
        input_data: str | list[dict],
        previous_response_id: str | None = None,
        max_tokens: int | None = None,
    ):
        """
        Send one request in a stored API conversation and return the full response.
        """
        if self.backend != "api":
            raise ValueError("Conversation calls require the api backend")

        request = self._build_api_request(
            input_data,
            max_tokens,
            conversation=True,
            previous_response_id=previous_response_id,
        )
        return self._create_api_response(request)

    def build_api_message(self, text: str, image_paths: list[Path]) -> list[dict]:
        """
        Build one user message with text followed by base64-encoded images.
        """
        import base64

        image_content = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            media_type = self.IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
            if not media_type:
                raise ValueError(f"Unsupported image type: {image_path.suffix}")

            image_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{encoded_image}",
                }
            )

        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    *image_content,
                ],
            }
        ]

    def call_llm(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a response from a text prompt.
        """
        if self.backend == "api":
            return self._call_api(prompt, max_tokens=max_tokens)

        sampling_params = self.sampling_params
        if max_tokens is not None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.local_temperature,
                max_tokens=max_tokens,
            )

        outputs = self.model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        return self._get_text(outputs)

    def call_vlm(self, prompt: str, folder_path: str | Path) -> str:
        """
        Generate a response from a prompt and all frames in a folder.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Frame folder not found: {folder}")

        image_paths = []
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                image_paths.append(path)

        image_paths.sort(key=parse_frame_timestamp)
        if not image_paths:
            raise ValueError(f"No image frames found in folder: {folder}")

        return self.call_vlm_images(prompt, image_paths)

    def call_vlm_images(
        self,
        prompt: str,
        image_paths: list[Path],
        include_frame_manifest: bool = True,
    ) -> str:
        """
        Generate a response from a prompt and an ordered list of frame images.

        Set include_frame_manifest=False for tasks that do not need frame
        filenames in the prompt or response.
        """
        if include_frame_manifest:
            # Keep original filenames available to tasks that return frame
            # references. Image payloads do not preserve local filenames.
            frame_manifest = "\n".join(
                f"Frame {index}: {image_path.name}"
                for index, image_path in enumerate(image_paths, start=1)
            )
            prompt_with_manifest = (
                f"{prompt}\n\n"
                "The images are provided in the same order as this frame filename "
                "manifest:\n"
                f"{frame_manifest}\n"
                "Use the exact filenames from this manifest in your response. "
                "Do not create replacement filenames."
            )
        else:
            prompt_with_manifest = prompt

        # ----------------------------------------------------
        # BACKEND: API
        # ----------------------------------------------------
        if self.backend == "api":
            return self._call_api(
                self.build_api_message(prompt_with_manifest, image_paths)
            )

        # ----------------------------------------------------
        # BACKEND: LOCAL (vLLM)
        # ----------------------------------------------------
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.copy())

        # 1. Structure the message matching Hugging Face / Qwen VL format
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt_with_manifest},
                ],
            }
        ]

        # 2. Inject official placeholder tokens and chat tags
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3. Pass formatted prompt alongside images to vLLM
        outputs = self.model.generate(
            {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": images},
            },
            sampling_params=self.sampling_params,
        )
        return self._get_text(outputs)
